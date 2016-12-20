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

SVMPtr trainSVM(const cv::Mat &data_,
				const cv::Mat &resp_)
{
	cv::SVMParams svmParams;
	svmParams.kernel_type = cv::SVM::RBF;
	svmParams.svm_type = cv::SVM::C_SVC;
	svmParams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10000, 1e-10);
	svmParams.C = 100;
	svmParams.gamma = 2;

	cv::Mat weights = cv::Mat::zeros(2, 1, CV_32FC1);
	weights.at<float>(0, 0) = 0.9;
	weights.at<float>(1, 0) = 0.1;
	// svmParams.class_weights = &weights.CvMat();

	// Train the SVM
	SVMPtr svm = SVMPtr(new cv::SVM());
	svm->train(data_, resp_, cv::Mat(), cv::Mat(), svmParams);
	// svm->train_auto(data_, resp_, cv::Mat(), cv::Mat(), svmParams);

	return svm;
}

void evalClassifier(const ModelPtr &model_,
					const cv::Mat &data_,
					const cv::Mat &resp_)
{
	bool isSVM = dynamic_cast<cv::SVM *>(model_.get()) != NULL;
	bool isBoost = dynamic_cast<cv::Boost *>(model_.get()) != NULL;
	bool isNetwork = dynamic_cast<cv::NeuralNet_MLP *>(model_.get()) != NULL;


	cv::Mat output;
	if (isSVM)
	{
		LOGD << "Predicting with SVM";
		cv::SVM *svm = dynamic_cast<cv::SVM *>(model_.get());
		svm->predict(data_, output);
	}
	if (isBoost)
	{
		LOGD << "Predicting with Boost";
		cv::Boost *boost = dynamic_cast<cv::Boost *>(model_.get());
		boost->predict(data_, output);
	}
	if (isNetwork)
	{
		LOGD << "Predicting with Neural Network";
		cv::NeuralNet_MLP *network = dynamic_cast<cv::NeuralNet_MLP *>(model_.get());
		network->predict(data_, output);
	}


	int tp = 0, tn = 0, fp = 0, fn = 0;
	for (int i = 0; i < output.rows; i++)
	{
		int prediction = output.at<float>(i);
		int response = resp_.at<float>(i);
		if (prediction == 0)
		{
			if (response == 0)
				tn++;
			else
				fn++;
		}
		else
		{
			if (response == 1)
				tp++;
			else
				fp++;
		}
	}
	int realn = tn + fp;
	int realp = fn + tp;
	int total = realn + realp;


	boost::format ff = boost::format("%.3f");
	std::ostringstream table;
	table << "\n";
	table << "-------------------------------------" << "\taccuracy\t: " << ff % (float(tp + tn) / float(total)) << "\n";
	table << "|         | pred: 0 | pred: 1 | SUM |" << "\tmissclass\t: " << ff % (float(fp + fn) / float(total)) << "\n";
	table << "-------------------------------------" << "\tTPR\t\t: " << ff % (float(tp) / float(realp)) << "\n";
	table << "| resp: 0 |   " << boost::format("%3d") % tn << "   |   " << boost::format("%3d") % fp << "   | " << boost::format("%3d") % realn << " |" << "\tFPR\t\t: " << ff % (float(fp) / float(realn)) << "\n";
	table << "| resp: 1 |   " << boost::format("%3d") % fn << "   |   " << boost::format("%3d") % tp << "   | " << boost::format("%3d") % realp << " |" << "\tspecificity\t: " << ff % (float(tn) / float(realn)) << "\n";
	table << "-------------------------------------" << "\tprecision\t: " << ff % (float(tp) / float(tp + fp)) << "\n";
	table << "|   SUM   |   " << boost::format("%3d") % (tn + fn) << "   |   " << boost::format("%3d") % (fp + tp) << "   | " << boost::format("%3d") % total << " |" << "\tprevalence\t: " << ff % (float(realp) / float(total)) << "\n";
	table << "-------------------------------------";

	LOGI << table.str();
}


int main(int _argn, char **_argv)
{
	static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
	plog::init(plog::severityFromString(YAML::LoadFile(LOGGING_LOCATION)["level"].as<std::string>().c_str()), &consoleAppender);


	clock_t begin = clock();
	try
	{
		// Check if enough arguments were given
		if (_argn < 2)
			throw std::runtime_error("Not enough params given\n\tUsage: Trainer <train_dir>");
		std::string trainDir = _argv[1];
		LOGI << "START!";


		LOGI << "Extracting train data";
		std::vector<std::vector<std::string> > tcsv;
		std::vector<std::vector<float> > train;
		std::vector<float> tresp;
		int successful = extractData(trainDir, tcsv, train, tresp);

		LOGI << "Traversed " << tcsv.size() << " files";
		LOGI << "\t- completed: " << train.size();
		LOGI << "\t- successful: " << successful;
		LOGI << "\t- ratio:: " << float(successful) / float(train.size());


		cv::Mat tdmat, trmat;
		prepareData(train, tresp, tdmat, trmat);

		SVMPtr svm = trainSVM(tdmat, trmat);
		evalClassifier(svm, tdmat, trmat);
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
