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
#include <fstream>


#define OUTPUT_DIR			"./output/"
#define CONFIG_DIR			"./config/"
#define LOGGING_LOCATION	CONFIG_DIR "logging.yaml"
#define CONFIG_LOCATION		CONFIG_DIR "config.yaml"


typedef boost::shared_ptr<cv::StatModel> ModelPtr;
typedef boost::shared_ptr<cv::SVM> SVMPtr;
typedef boost::shared_ptr<cv::Boost> BoostingPtr;
typedef boost::shared_ptr<cv::NeuralNet_MLP> NeuralNetworkPtr;


YAML::Node config;


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
		if (config["useAngle"].as<bool>())
			data_.back().push_back(file_["orientation"]["angle"].as<float>());

		response_.push_back(file_["result"]["success"].as<bool>());
	}
}

/**************************************************/
std::vector<std::string> genCSVHeader()
{
	std::vector<std::string> header;
	header.push_back("object");
	header.push_back("completed");
	header.push_back("successful");
	header.push_back("err_code");
	header.push_back("cluster");
	header.push_back("angle");
	header.push_back("splits");
	header.push_back("grasp_id");
	header.push_back("experiment");
	header.push_back("data_set");
	header.push_back("descriptor");
	return header;
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
				const cv::Mat &resp_,
				const float ratio_,
				const bool auto_ = false)
{
	cv::SVMParams params;
	params.svm_type = config["svm"]["svm_type"].as<int>();
	params.kernel_type = config["svm"]["kernel_type"].as<int>();
	params.degree = config["svm"]["degree"].as<float>();
	params.gamma = config["svm"]["gamma"].as<float>();
	params.coef0 = config["svm"]["coef0"].as<float>();

	params.C = config["svm"]["C"].as<float>();
	params.nu = config["svm"]["nu"].as<float>();
	params.p = config["svm"]["p"].as<float>();

	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10000, 1e-10);


	if (config["svm"]["useWeights"].as<bool>())
	{
		cv::Mat aux = cv::Mat::zeros(2, 1, CV_32FC1);
		aux.at<float>(0, 0) = 0.5 / (1 - ratio_);
		aux.at<float>(1, 0) = 0.5 / ratio_;
		CvMat weights = cvMat(2, 1, CV_32FC1, aux.data);
		params.class_weights = &weights;
	}


	SVMPtr svm = SVMPtr(new cv::SVM());
	if (!auto_)
		svm->train(data_, resp_, cv::Mat(), cv::Mat(), params);
	else
		svm->train_auto(data_, resp_, cv::Mat(), cv::Mat(), params);

	return svm;
}

/**************************************************/
BoostingPtr trainBoost(const cv::Mat &data_,
					   const cv::Mat &resp_,
					   const float ratio_)
{
	cv::BoostParams params;
	params.boost_type = config["boost"]["boost_type"].as<int>();
	params.weak_count = config["boost"]["weak_count"].as<int>();
	params.split_criteria = config["boost"]["split_criteria"].as<int>();
	params.weight_trim_rate = config["boost"]["weight_trim_rate"].as<float>();
	params.cv_folds = config["boost"]["cv_folds"].as<int>();


	BoostingPtr boost = BoostingPtr(new cv::Boost());
	boost->train(data_, CV_ROW_SAMPLE, resp_, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), params);

	return boost;
}

/**************************************************/
NeuralNetworkPtr trainNetwork(const cv::Mat &data_,
							  const cv::Mat &resp_,
							  const float ratio_)
{
	cv::ANN_MLP_TrainParams params;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 1e-6);
	params.train_method = config["network"]["train_method"].as<int>();

	// Back propagation params
	params.bp_dw_scale = config["network"]["bp_dw_scale"].as<float>();
	params.bp_moment_scale = config["network"]["bp_moment_scale"].as<float>();

	// R propagation params
	params.rp_dw0 = config["network"]["rp_dw0"].as<float>();
	params.rp_dw_plus = config["network"]["rp_dw_plus"].as<float>();
	params.rp_dw_minus = config["network"]["rp_dw_minus"].as<float>();
	params.rp_dw_min = config["network"]["rp_dw_min"].as<float>();
	params.rp_dw_max = config["network"]["rp_dw_max"].as<float>();


	std::vector<int> l = config["network"]["layers"].as<std::vector<int> >();
	cv::Mat layers = cv::Mat::zeros(1 + l.size() + 1, 1, CV_32SC1);

	size_t i = 0;
	layers.at<int>(i++) = data_.cols;
	for (; i - 1 < l.size(); i++)
		layers.at<int>(i) = l[i - 1];
	layers.at<int>(i) = 1;
	NeuralNetworkPtr network = NeuralNetworkPtr(new cv::NeuralNet_MLP());

	int activate_fn = config["network"]["activate_fn"].as<int>();
	float alpha = config["network"]["alpha"].as<float>();
	float beta = config["network"]["beta"].as<float>();
	network->create(layers, activate_fn, alpha, beta);


	cv::Mat weights;
	// if (params.train_method == cv::ANN_MLP_TrainParams::RPROP)
	// {
	// 	weights = cv::Mat::zeros(resp_.rows, 1, CV_32FC1);
	// 	for (int i = 0; i < resp_.rows; i++)
	// 	{
	// 		float label = resp_.at<float>(i, 0);
	// 		// weights.at<float>(i, 0) = fabs(label - 1) < 1e-7 ? (0.5 / ratio_) : (0.5 / (1 - ratio_));
	// 		weights.at<float>(i, 0) = fabs(label - 1) < 1e-7 ? (0.5 / ratio_) : 1;
	// 		LOGI << resp_.at<float>(i, 0) << " -- " << weights.at<float>(i, 0);
	// 	}
	// }


	network->train(data_, resp_, weights, cv::Mat(), params);

	return network;
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

		bool show = config["svm"]["showPredictions"].as<bool>();
		if (show)
		{
			LOGD << "sup vectors: " << svm->get_support_vector_count();
			for (int i = 0; show && i < vdata_.rows; i++)
			{
				float distance = svm->predict(vdata_.row(i), true);
				int label = svm->predict(vdata_.row(i), false);
				LOGD << "resp: " << vresp_.row(i) << " - label: " << label << " - dist: " << distance;
			}
		}
	}
	else if (isBoost)
	{
		LOGD << "Predicting with Boost";
		cv::Boost *boost = dynamic_cast<cv::Boost *>(model_.get());

		bool show = config["boost"]["showPredictions"].as<bool>();
		if (show)
		{
			for (int i = 0; show && i < vdata_.rows; i++)
			{
				float votes = boost->predict(vdata_.row(i), cv::Mat(), cv::Range::all(), false, true);
				float label = boost->predict(vdata_.row(i), cv::Mat(), cv::Range::all(), false, false);
				LOGD << "resp: " << vresp_.row(i) << " - label: " << label << " - votes: " << votes;
			}
		}

		tout = cv::Mat::zeros(tdata_.rows, 1, CV_32FC1);
		for (int i = 0; i < tdata_.rows; i++)
			tout.at<float>(i, 0) = boost->predict(tdata_.row(i));

		vout = cv::Mat::zeros(vdata_.rows, 1, CV_32FC1);
		for (int i = 0; i < vdata_.rows; i++)
			vout.at<float>(i, 0) = boost->predict(vdata_.row(i));
	}
	else if (isNetwork)
	{
		LOGD << "Predicting with Neural Network";
		cv::NeuralNet_MLP *network = dynamic_cast<cv::NeuralNet_MLP *>(model_.get());

		network->predict(tdata_, tout);
		for (int i = 0; i < tout.rows; i++)
			tout.at<float>(i, 0) = tout.at<float>(i, 0) > 0 ? 1 : 0;

		network->predict(vdata_, vout);
		for (int i = 0; i < vout.rows; i++)
			vout.at<float>(i, 0) = vout.at<float>(i, 0) > 0 ? 1 : 0;
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


	std::string sp = "  ";
	boost::format ff = boost::format("%.3f");
	std::vector<std::string> stats;
	stats.push_back("\t\t  TRAIN" + sp + " VAL");
	stats.push_back("Accuracy\t: "
					+ boost::str(ff % (float(t_tp + t_tn) / t_total)) + sp
					+ boost::str(ff % (float(v_tp + v_tn) / v_total))
					+ "  [tp + tn / total]");
	stats.push_back("Miss\t: "
					+ boost::str(ff % (float(t_fp + t_fn) / t_total)) + sp
					+ boost::str(ff % (float(v_fp + v_fn) / v_total))
					+ "  [fp + fn / total]");
	stats.push_back("TP rate\t: "
					+ boost::str(ff % (float(t_tp) / t_realp)) + sp
					+ boost::str(ff % (float(v_tp) / v_realp))
					+ "  [tp / realp]");
	stats.push_back("FP rate\t: "
					+ boost::str(ff % (float(t_fp) / t_realn)) + sp
					+ boost::str(ff % (float(v_fp) / v_realn))
					+ "  [fp / realn]");
	stats.push_back("Specificity\t: "
					+ boost::str(ff % (float(t_tn) / t_realn)) + sp
					+ boost::str(ff % (float(v_tn) / v_realn))
					+ "  [tn / realn]");
	stats.push_back("Precision\t: "
					+ boost::str(ff % (float(t_tp) / (t_tp + t_fp))) + sp
					+ boost::str(ff % (float(v_tp) / (v_tp + v_fp)))
					+ "  [tp / tp + fp]");
	stats.push_back("Prevalence\t: "
					+ boost::str(ff % (float(t_realp) / t_total)) + sp
					+ boost::str(ff % (float(v_realp) / v_total))
					+ "  [realp / total]");


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
			ss << "    " << stats[i];
		ss << "\n";
	}

	LOGI << ss.str();
}

/**************************************************/
std::string getId(const std::string &dir_)
{
	std::string id = dir_;
	if (*(id.end() - 1) == '/' || *(id.end() - 1) == '\\')
		id = id.erase(id.length() - 1);

	size_t index = id.find_last_of('/');
	if (index == std::string::npos)
		index = id.find_last_of('\\');

	if (index != std::string::npos)
		id = id.substr(index + 1);

	return id;
}

/**************************************************/
void generateCSV(const std::string &basename_,
				 const std::vector<std::string> &header_,
				 const std::vector<std::vector<std::string> > &data_,
				 const std::vector<std::vector<float> > &descriptor_,
				 const std::vector<float> &response_)
{
	std::ofstream csvData;
	csvData.open((OUTPUT_DIR + basename_ + "_data.csv").c_str(), std::fstream::out);

	for (size_t i = 0; i < header_.size(); i++)
		csvData << header_[i] << ",";
	csvData << "\n";

	for (size_t i = 0; i < data_.size(); i++)
	{
		for (size_t j = 0; j < data_[i].size(); j++)
			csvData << data_[i][j] << ",";
		csvData << "\n";
	}

	csvData.close();


	std::ofstream csvDesc;
	csvDesc.open((OUTPUT_DIR + basename_ + "_desc.csv").c_str(), std::fstream::out);

	for (size_t i = 0; i < descriptor_.size(); i++)
	{
		csvDesc << response_[i] << ",";
		for (size_t j = 0; j < descriptor_[i].size(); j++)
			csvDesc << descriptor_[i][j] << ",";
		csvDesc << "\n";
	}

	csvDesc.close();
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
		config = YAML::LoadFile(CONFIG_LOCATION);


		if (system("mkdir -p " OUTPUT_DIR) != 0)
			throw std::runtime_error("Can't create output directory");
		std::string id = getId(trainDir);


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


		if (config["trainSVM"].as<bool>())
		{
			LOGI << "*** Training SVM ***";
			SVMPtr svm = trainSVM(tdmat, trmat, tratio);
			SVMPtr svm_auto = trainSVM(tdmat, trmat, tratio, true);

			LOGI << "Evaluating SVM";
			evalClassifier(svm, tdmat, trmat, vdmat, vrmat);
			LOGI << "Evaluating SVM-auto";
			evalClassifier(svm_auto, tdmat, trmat, vdmat, vrmat);

			svm->save((OUTPUT_DIR + id + "_svm.yaml").c_str());
			svm_auto->save((OUTPUT_DIR  + id + "_svm_auto.yaml").c_str());
		}


		if (config["trainBoost"].as<bool>())
		{
			LOGI << "*** Training Boost ***";
			BoostingPtr boost = trainBoost(tdmat, trmat, tratio);

			LOGI << "Evaluating Boost";
			evalClassifier(boost, tdmat, trmat, vdmat, vrmat);

			boost->save((OUTPUT_DIR  + id + "_boost.yaml").c_str());
		}


		if (config["trainNetwork"].as<bool>())
		{
			LOGI << "*** Training Network ***";
			NeuralNetworkPtr network = trainNetwork(tdmat, trmat, tratio);

			LOGI << "Evaluating Network";
			evalClassifier(network, tdmat, trmat, vdmat, vrmat);

			network->save((OUTPUT_DIR  + id + "_network.yaml").c_str());
		}


		std::vector<std::string> header = genCSVHeader();
		generateCSV(id + "_train", header, tcsv, train, tresp);
		generateCSV(id + "_val", header, vcsv, val, vresp);
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
