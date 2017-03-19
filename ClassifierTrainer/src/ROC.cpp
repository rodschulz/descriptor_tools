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


typedef boost::shared_ptr<cv::StatModel> ModelPtr;
typedef boost::shared_ptr<cv::SVM> SVMPtr;
typedef boost::shared_ptr<cv::Boost> BoostingPtr;
typedef boost::shared_ptr<cv::NeuralNet_MLP> NeuralNetworkPtr;


/**************************************************/
// void prepareData(const std::vector<std::vector<float> > &data_,
// 				 const std::vector<float> &resp_,
// 				 cv::Mat &cvData_,
// 				 cv::Mat &cvResp_)
// {
// 	// Copy data array
// 	if (cvData_.rows != (int)data_.size())
// 		cvData_ = cv::Mat::zeros(data_.size(), data_[0].size(), CV_32FC1);

// 	for (size_t i = 0; i < data_.size(); i++)
// 	{
// 		if (data_[i].size() != (size_t) cvData_.cols)
// 			throw std::runtime_error("Mismatching descriptor sizes (" + boost::lexical_cast<std::string>(data_[i].size()) + " != " + boost::lexical_cast<std::string>(cvData_.cols) + ")");

// 		memcpy(&cvData_.at<float>(i, 0), &data_[i][0], sizeof(float) * cvData_.cols);
// 	}


// 	// Copy response array
// 	if (cvResp_.rows != (int)resp_.size())
// 		cvResp_ = cv::Mat::zeros(resp_.size(), 1, CV_32FC1);

// 	for (size_t i = 0; i < resp_.size(); i++)
// 		cvResp_.at<float>(i, 0) = resp_[i];
// }

/**************************************************/
void evalClassifier(const ModelPtr &model_,
					const cv::Mat &tdata_,
					const cv::Mat &tresp_,
					const cv::Mat &vdata_,
					const cv::Mat &vresp_)
{
	// bool isSVM = dynamic_cast<cv::SVM *>(model_.get()) != NULL;
	// bool isBoost = dynamic_cast<cv::Boost *>(model_.get()) != NULL;
	// bool isNetwork = dynamic_cast<cv::NeuralNet_MLP *>(model_.get()) != NULL;


	// cv::Mat tout, vout;
	// if (isSVM)
	// {
	// 	LOGD << "Predicting with SVM";
	// 	cv::SVM *svm = dynamic_cast<cv::SVM *>(model_.get());

	// 	svm->predict(tdata_, tout);
	// 	svm->predict(vdata_, vout);

	// 	bool show = config["svm"]["showPredictions"].as<bool>();
	// 	if (show)
	// 	{
	// 		LOGD << "sup vectors: " << svm->get_support_vector_count();
	// 		for (int i = 0; show && i < vdata_.rows; i++)
	// 		{
	// 			float distance = svm->predict(vdata_.row(i), true);
	// 			int label = svm->predict(vdata_.row(i), false);
	// 			LOGD << "resp: " << vresp_.row(i) << " - label: " << label << " - dist: " << distance;
	// 		}
	// 	}
	// }
	// else if (isBoost)
	// {
	// 	LOGD << "Predicting with Boost";
	// 	cv::Boost *boost = dynamic_cast<cv::Boost *>(model_.get());

	// 	tout = cv::Mat::zeros(tdata_.rows, 1, CV_32FC1);
	// 	for (int i = 0; i < tdata_.rows; i++)
	// 		tout.at<float>(i, 0) = boost->predict(tdata_.row(i));

	// 	vout = cv::Mat::zeros(vdata_.rows, 1, CV_32FC1);
	// 	for (int i = 0; i < vdata_.rows; i++)
	// 		vout.at<float>(i, 0) = boost->predict(vdata_.row(i));
	// }
	// else if (isNetwork)
	// {
	// 	LOGD << "Predicting with Neural Network";
	// 	cv::NeuralNet_MLP *network = dynamic_cast<cv::NeuralNet_MLP *>(model_.get());

	// 	network->predict(tdata_, tout);
	// 	for (int i = 0; i < tout.rows; i++)
	// 		tout.at<float>(i, 0) = tout.at<float>(i, 0) > 0 ? 1 : 0;

	// 	network->predict(vdata_, vout);
	// 	for (int i = 0; i < vout.rows; i++)
	// 		vout.at<float>(i, 0) = vout.at<float>(i, 0) > 0 ? 1 : 0;
	// }


	// int t_tp = 0, t_tn = 0, t_fp = 0, t_fn = 0;
	// countCases(tout, tresp_, t_tp, t_tn, t_fp, t_fn);
	// int t_realn = t_tn + t_fp;
	// int t_realp = t_fn + t_tp;
	// int t_total = t_realn + t_realp;


	// int v_tp = 0, v_tn = 0, v_fp = 0, v_fn = 0;
	// countCases(vout, vresp_, v_tp, v_tn, v_fp, v_fn);
	// int v_realn = v_tn + v_fp;
	// int v_realp = v_fn + v_tp;
	// int v_total = v_realn + v_realp;
}

/**************************************************/
std::pair<cv::Mat, cv::Mat> readCSV(const std::string &filename_)
{
	std::vector<float> response;
	std::vector<std::vector<float> > descriptor;

	std::string line;
	std::ifstream file;
	file.open(filename_.c_str(), std::fstream::in);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			std::stringstream lineStream(line);
			std::string cell;

			int read = 0;
			while (std::getline(lineStream, cell, ','))
			{
				if ((read++) == 0)
				{
					response.push_back(boost::lexical_cast<float>(cell));
					descriptor.push_back(std::vector<float>());
				}
				else
					descriptor.back().push_back(boost::lexical_cast<float>(cell));
			}
		}
	}

	LOGD << ".......loaded " << response.size() << "/" << descriptor.size() << " lines";

	std::pair<cv::Mat, cv::Mat> data;
	data.first = cv::Mat::zeros(response.size(), 1, CV_32FC1);
	memcpy(data.first.data, &response[0], response.size() * sizeof(float));

	data.second = cv::Mat::zeros(descriptor.size(), descriptor[0].size(), CV_32FC1);
	for (int row = 0; row < data.second.rows; row++)
		for (int col = 0; col < data.second.cols; col++)
			data.second.at<float>(row, col) = descriptor[row][col];

	return data;
}

/**************************************************/
void parseArgs(const int argn_,
			   const char **argv_,
			   std::vector<ModelPtr> &models_,
			   std::vector<std::pair<cv::Mat, cv::Mat> > &data_)
{
	for (int i = 1; i < argn_; i += 2)
	{
		// Load the model
		LOGD << "...loading model";
		YAML::Node model = YAML::LoadFile(argv_[i]);
		if (model["my_svm"])
			models_.push_back(ModelPtr(new cv::SVM()));
		else if (model["my_boost_tree"])
			models_.push_back(ModelPtr(new cv::Boost()));
		else if (model["my_nn"])
			models_.push_back(ModelPtr(new cv::NeuralNet_MLP()));

		models_.back()->load(argv_[i]);
		LOGD << ".....type loaded: " << typeid(*models_.back()).name();

		LOGD << "...loading csv";
		data_.push_back(readCSV(argv_[i + 1]));
	}
}

/**************************************************/
int main(const int argn_, const char **argv_)
{
	static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
	plog::init(plog::severityFromString(YAML::LoadFile(LOGGING_LOCATION)["level"].as<std::string>().c_str()), &consoleAppender);

	clock_t begin = clock();
	try
	{
		// Check if enough arguments were given
		if (argn_ < 3)
			throw std::runtime_error("Not enough params given\n\tUsage: ROC <model_1> <val_csv_1> ... <model_N> <val_csv_N>");

		LOGI << "START!";

		if (system("mkdir -p " OUTPUT_DIR) != 0)
			throw std::runtime_error("Can't create output directory");

		LOGI << "Parsing arguments";
		std::vector<ModelPtr> models;
		std::vector<std::pair<cv::Mat, cv::Mat> > data;
		parseArgs(argn_, argv_, models, data);



		// std::pair<cv::Mat, cv::Mat> xx = readCSV("./output/DCH_72_split4_beer_drill_train_desc.csv");
		// cv::Mat img = xx.second;

		// LOGI << "size img " << img.rows << " -- " << img.cols;

		// for (int row = 0; row < img.rows; row++)
		// {
		// 	for (int col = 0; col < img.cols; col++)
		// 	{
		// 		std::cout << img.at<float>(row, col) << "\t";
		// 	}
		// 	std::cout << "\n";
		// }



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
