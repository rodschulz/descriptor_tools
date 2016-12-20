/**
 * Author: rodrigo
 * 2016
 */
#include <plog/Log.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>


#define LOGGING_LOCATION "./logging.yaml"


typedef boost::shared_ptr<cv::StatModel> ModelPtr;
typedef boost::shared_ptr<cv::SVM> SVMPtr;
typedef boost::shared_ptr<cv::Boost> BoostingPtr;
typedef boost::shared_ptr<cv::NeuralNet_MLP> NeuralNetworkPtr;


/**************************************************/
void loadFromFile(const YAML::Node file_,
				  std::vector<std::string> &csv_,
				  std::vector<std::vector<float> > &data_,
				  std::vector<float> &response_)
{
	csv_.push_back(file_["target_object"].as<std::string>());

	csv_.push_back(file_["result"]["attempt_completed"].as<std::string>());
	csv_.push_back(file_["result"]["success"].as<std::string>());
	csv_.push_back(file_["result"]["pick_error_code"].as<std::string>());

	csv_.push_back(file_["cluster_label"].as<std::string>());

	csv_.push_back(file_["orientation"]["angle"].as<std::string>());
	csv_.push_back(file_["orientation"]["split_number"].as<std::string>());

	csv_.push_back(file_["grasp"]["id"].as<std::string>());

	if (file_["result"]["attempt_completed"].as<bool>())
	{
		data_.push_back(file_["descriptor"]["data"].as<std::vector<float> >());
		data_.back().push_back(file_["orientation"]["angle"].as<float>());

		response_.push_back(file_["result"]["success"].as<bool>());
	}
}

/**************************************************/
int extractData(const std::string &directory_,
				std::vector<std::vector<std::string> > &csv_,
				std::vector<std::vector<float> > &data_,
				std::vector<float> &response_)
{
	int successfull = 0;

	boost::filesystem::path target(directory_);
	boost::filesystem::directory_iterator it(target), eod;
	BOOST_FOREACH(boost::filesystem::path const & filepath, std::make_pair(it, eod))
	{
		if (is_regular_file(filepath))
		{
			if (!boost::iequals(filepath.extension().string(), ".yaml"))
				continue;

			LOGD << "Processing " << filepath.filename();
			YAML::Node file =  YAML::LoadFile(filepath.string().c_str());
			csv_.push_back(std::vector<std::string>());
			loadFromFile(file, csv_.back(), data_, response_);

			// Add experiment and set
			csv_.back().push_back(filepath.filename().string());

			if (boost::iequals(csv_.back()[2], "true"))
				successfull++;

			std::string set = filepath.string();
			set = set.substr(0, set.find_last_of('/'));
			set = set.substr(set.find_last_of('/') + 1);
			csv_.back().push_back(set);
		}
		else
			extractData(filepath.string(), csv_, data_, response_);
	}

	return successfull;
}

/**************************************************/
void prepareData(const std::vector<std::vector<float> > &data_,
				 const std::vector<float> &resp_,
				 cv::Mat &cvData_,
				 cv::Mat &cvResp_)
{
	// Copy data array
	if (cvData_.rows != (int)data_.size())
		cvData_ = cv::Mat::zeros(data_.size(), data_[0].size(), CV_32FC1);

	for (size_t i = 0; i < data_.size(); i++)
	{
		if (data_[i].size() != (size_t) cvData_.cols)
			throw std::runtime_error("Mismatching descriptor sizes (" + boost::lexical_cast<std::string>(data_[i].size()) + " != " + boost::lexical_cast<std::string>(cvData_.cols) + ")");

		memcpy(&cvData_.at<float>(i, 0), &data_[i][0], sizeof(float) * cvData_.cols);
	}


	// Copy response array
	if (cvResp_.rows != (int)resp_.size())
		cvResp_ = cv::Mat::zeros(resp_.size(), 1, CV_32FC1);

	for (size_t i = 0; i < resp_.size(); i++)
		cvResp_.at<float>(i, 0) = resp_[i];
}

/**************************************************/
SVMPtr trainSVM(const cv::Mat &data_,
				const cv::Mat &resp_)
{
	cv::SVMParams svmParams;
	svmParams.kernel_type = cv::SVM::RBF;
	svmParams.svm_type = cv::SVM::C_SVC;
	svmParams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10000, 1e-10);
	svmParams.C = 100;
	svmParams.gamma = 2;

	cv::Mat aux = cv::Mat::zeros(2, 1, CV_32FC1);
	aux.at<float>(0, 0) = 1;
	aux.at<float>(1, 0) = 1 / 0.303;
	CvMat weights = cvMat(2, 1, CV_32FC1, aux.data);
	svmParams.class_weights = &weights;

	// Train the SVM
	SVMPtr svm = SVMPtr(new cv::SVM());
	svm->train(data_, resp_, cv::Mat(), cv::Mat(), svmParams);
	// svm->train_auto(data_, resp_, cv::Mat(), cv::Mat(), svmParams);

	return svm;
}

/**************************************************/
void countCases(const cv::Mat &out_,
				const cv::Mat &resp_,
				int &tp_,
				int &tn_,
				int &fp_,
				int &fn_)
{
	for (int i = 0; i < out_.rows; i++)
	{
		int prediction = out_.at<float>(i);
		int response = resp_.at<float>(i);
		if (prediction == 0)
		{
			if (response == 0)
				tn_++;
			else // response == 1
				fn_++;
		}
		else // prediction == 1
		{
			if (response == 1)
				tp_++;
			else // response == 0
				fp_++;
		}
	}
}

/**************************************************/
void evalClassifier(const ModelPtr &model_,
					const cv::Mat &tdata_,
					const cv::Mat &tresp_,
					const cv::Mat &vdata_,
					const cv::Mat &vresp_)
{
	bool isSVM = dynamic_cast<cv::SVM *>(model_.get()) != NULL;
	bool isBoost = dynamic_cast<cv::Boost *>(model_.get()) != NULL;
	bool isNetwork = dynamic_cast<cv::NeuralNet_MLP *>(model_.get()) != NULL;


	cv::Mat tout, vout;
	if (isSVM)
	{
		LOGD << "Predicting with SVM";
		cv::SVM *svm = dynamic_cast<cv::SVM *>(model_.get());
		svm->predict(tdata_, tout);
		svm->predict(vdata_, vout);
	}
	else if (isBoost)
	{
		LOGD << "Predicting with Boost";
		cv::Boost *boost = dynamic_cast<cv::Boost *>(model_.get());
		boost->predict(tdata_, tout);
	}
	else if (isNetwork)
	{
		LOGD << "Predicting with Neural Network";
		cv::NeuralNet_MLP *network = dynamic_cast<cv::NeuralNet_MLP *>(model_.get());
		network->predict(tdata_, tout);
	}


	int t_tp = 0, t_tn = 0, t_fp = 0, t_fn = 0;
	countCases(tout, tresp_, t_tp, t_tn, t_fp, t_fn);
	int t_realn = t_tn + t_fp;
	int t_realp = t_fn + t_tp;
	int t_total = t_realn + t_realp;


	int v_tp = 0, v_tn = 0, v_fp = 0, v_fn = 0;
	countCases(vout, vresp_, v_tp, v_tn, v_fp, v_fn);
	int v_realn = v_tn + v_fp;
	int v_realp = v_fn + v_tp;
	int v_total = v_realn + v_realp;


	boost::format fd = boost::format("%3d");
	std::vector<std::string> table;
	table.push_back("----------------------------------------------------------------");
	table.push_back("|         |          TRAIN          ||       VALIDATION        |");
	table.push_back("----------------------------------------------------------------");
	table.push_back("|         | pred: 0 | pred: 1 | SUM || pred: 0 | pred: 1 | SUM |");
	table.push_back("----------------------------------------------------------------");
	table.push_back("| resp: 0 |   "
					+ boost::str(fd % t_tn) + "   |   "
					+ boost::str(fd % t_fp) + "   | "
					+ boost::str(fd % t_realn) + " ||   "
					+ boost::str(fd % v_tn) + "   |   "
					+ boost::str(fd % v_fp) + "   | "
					+ boost::str(fd % v_realn) + " |");
	table.push_back("| resp: 1 |   "
					+ boost::str(fd % t_fn) + "   |   "
					+ boost::str(fd % t_tp) + "   | "
					+ boost::str(fd % t_realp) + " ||   "
					+ boost::str(fd % v_fn) + "   |   "
					+ boost::str(fd % v_tp) + "   | "
					+ boost::str(fd % v_realp) + " |");
	table.push_back("----------------------------------------------------------------");
	table.push_back("|   SUM   |   "
					+ boost::str(fd % (t_tn + t_fn)) + "   |   "
					+ boost::str(fd % (t_tp + t_fp)) + "   | "
					+ boost::str(fd % t_total) + " ||   "
					+ boost::str(fd % (v_tn + v_fn)) + "   |   "
					+ boost::str(fd % (v_tp + v_fp)) + "   | "
					+ boost::str(fd % v_total) + " |");
	table.push_back("----------------------------------------------------------------");


	std::string sp = "\t  ";
	boost::format ff = boost::format("%.3f");
	std::vector<std::string> stats;
	stats.push_back("\t\t  TRAIN" + sp + " VAL");
	stats.push_back("Accuracy\t: "
					+ boost::str(ff % (float(t_tp + t_tn) / t_total)) + sp
					+ boost::str(ff % (float(v_tp + v_tn) / v_total)));
	stats.push_back("Missclass\t: "
					+ boost::str(ff % (float(t_fp + t_fn) / t_total)) + sp
					+ boost::str(ff % (float(v_fp + v_fn) / v_total)));
	stats.push_back("TPR\t\t: "
					+ boost::str(ff % (float(t_tp) / t_realp)) + sp
					+ boost::str(ff % (float(v_tp) / v_realp)));
	stats.push_back("FPR\t\t: "
					+ boost::str(ff % (float(t_fp) / t_realn)) + sp
					+ boost::str(ff % (float(v_fp) / v_realn)));
	stats.push_back("Specificity\t: "
					+ boost::str(ff % (float(t_tn) / t_realn)) + sp
					+ boost::str(ff % (float(v_tn) / v_realn)));
	stats.push_back("Precision\t: "
					+ boost::str(ff % (float(t_tp) / (t_tp + t_fp))) + sp
					+ boost::str(ff % (float(v_tp) / (v_tp + v_fp))));
	stats.push_back("Prevalence\t: "
					+ boost::str(ff % (float(t_realp) / t_total)) + sp
					+ boost::str(ff % (float(v_realp) / v_total)));


	size_t tableSize = table.size();
	size_t statsSize = stats.size();
	size_t len = std::max(tableSize, statsSize);
	std::stringstream ss;
	ss << "\n";
	for (size_t i = 0; i < len; i++)
	{
		if (i < tableSize)
			ss << table[i];
		if (i < statsSize)
			ss << "\t" << stats[i];
		ss << "\n";
	}

	LOGI << ss.str();
}

/**************************************************/
int main(int _argn, char **_argv)
{
	static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
	plog::init(plog::severityFromString(YAML::LoadFile(LOGGING_LOCATION)["level"].as<std::string>().c_str()), &consoleAppender);


	clock_t begin = clock();
	try
	{
		// Check if enough arguments were given
		if (_argn < 3)
			throw std::runtime_error("Not enough params given\n\tUsage: Trainer <train_dir> <val_dir>");
		std::string trainDir = _argv[1];
		std::string valDir = _argv[2];
		LOGI << "START!";


		LOGI << "=== Extracting train data ===";
		std::vector<std::vector<std::string> > tcsv;
		std::vector<std::vector<float> > train;
		std::vector<float> tresp;
		int tsuccessful = extractData(trainDir, tcsv, train, tresp);
		float tratio = tsuccessful / float(train.size());

		LOGI << "Traversed " << tcsv.size() << " files";
		LOGI << "\t  - completed\t: " << train.size();
		LOGI << "\t  - successful\t: " << tsuccessful;
		LOGI << "\t  - ratio\t: " << tratio;


		LOGI << "=== Extracting validation data ===";
		std::vector<std::vector<std::string> > vcsv;
		std::vector<std::vector<float> > val;
		std::vector<float> vresp;
		int vsuccessful = extractData(valDir, vcsv, val, vresp);
		float vratio = vsuccessful / float(val.size());

		LOGI << "Traversed " << vcsv.size() << " files";
		LOGI << "\t  - completed\t: " << val.size();
		LOGI << "\t  - successful\t: " << vsuccessful;
		LOGI << "\t  - ratio\t: " << vratio;


		LOGI << "Preparing data";
		cv::Mat tdmat, trmat;
		prepareData(train, tresp, tdmat, trmat);
		cv::Mat vdmat, vrmat;
		prepareData(val, vresp, vdmat, vrmat);


		LOGI << "Training classifiers";
		SVMPtr svm = trainSVM(tdmat, trmat);


		LOGI << "*** Evaluating SVM ***";
		evalClassifier(svm, tdmat, trmat, vdmat, vrmat);
	}
	catch (std::exception &_ex)
	{
		LOGE << _ex.what();
	}

	clock_t end = clock();
	double elapsedTime = double(end - begin) / CLOCKS_PER_SEC;
	LOGI << std::fixed << std::setprecision(3) << "Finished in " << elapsedTime << " [s]";

	return EXIT_SUCCESS;
}
