// HaralickBasedDirectionalityMap.cpp : Defines the entry point for the console application.
// HaralickDirectionality.cpp : Defines the entry point for the console application.
//

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
//#include "opencv2\contrib\contrib.hpp"

#include <boost\filesystem.hpp>
#include <boost/regex.hpp>

#include <chrono>


//#include "math.h"
#include <iostream>
#include <fstream>
#//include <Windows.h>

#include "..\..\..\ProjectsLib\LibMarcin\Functions.h"
//#include "RedundantWaveletLib.h"
#include "..\..\..\ProjectsLib\LibMarcin\NormalizationLib.h"
#include "..\..\..\ProjectsLib\LibMarcin\HaralickLib.h"
#include "..\..\..\ProjectsLib\LibMarcin\ParamFromXML.h"
#include "..\..\..\ProjectsLib\LibMarcin\DispLib.h"
#include "..\..\..\ProjectsLib\LibMarcin\StringFcLib.h"
#include "..\..\..\ProjectsLib\LibMarcin\RegionU16Lib.h"

#include "..\..\..\ProjectsLib\tinyxml\tinyxml.h"
#include "..\..\..\ProjectsLib\tinyxml\tinystr.h"

#define PI 3.14159265

using namespace cv;
using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace std::chrono;


//const int stepNr = 180;


int main(int argc, char* argv[])
{
	// Only 1 argument: xml filename
	if (argc < 2)
	{
		cout << "\nTo few arguments.";
		return 0;
	}

	// Path is a structure from Boost file system
	path ConfigFile(argv[1]);
	if (!exists(ConfigFile))
	{
		cout << ConfigFile.filename().string() << " not exists " << '\n';
		return 0;
	}
	// ProcessOptions is a class that contains all necessary parameters
	ProcessOptions ProcOptions;
	// Read parameters from the XML file
	ProcOptions.LoadParams(ConfigFile.string());
	string ProcOptionsStr = ProcOptions.ShowParams();
	cout << ProcOptionsStr;

	path PathToProcess(ProcOptions.InFolderName1);
	if (!exists(PathToProcess))
	{
		cout << PathToProcess << " not exists " << '\n';
		return 0;
	}
	if (!is_directory(PathToProcess))
	{
		cout << PathToProcess << " This is not a directory path " << '\n';
		//return 0;
	}

	if (ProcOptions.displayResult)
	{
		namedWindow("Image", WINDOW_AUTOSIZE);
	}

	if (ProcOptions.displaySmallImage)
	{
		namedWindow("ImageSmall", WINDOW_AUTOSIZE);
	}

	regex FilePattern(ProcOptions.InFilePattern1);

	// Create ROI. The ROI could be the whole image or a tile of specific size and shape
	Mat Roi;
	int roiMaxX, roiMaxY; // Bounding box sizes for ROI 
	switch (ProcOptions.tileShape) // Different tile shapes
	{
	case 1: // Rectangle
		roiMaxX = ProcOptions.maxTileX;
		roiMaxY = ProcOptions.maxTileY;
		Roi = Mat::ones(roiMaxY, roiMaxX, CV_16U);
		break;
	case 2: // Ellipse
		roiMaxX = ProcOptions.maxTileX;
		roiMaxY = ProcOptions.maxTileY;
		Roi = Mat::zeros(roiMaxY, roiMaxX, CV_16U);
		ellipse(Roi, Point(roiMaxX / 2, roiMaxY / 2),
			Size(roiMaxX / 2, roiMaxY / 2), 0.0, 0.0, 360.0,
			1, -1);
		break;
	case 3: // Hexagon
	{
		int edgeLength = ProcOptions.maxTileX;
		roiMaxX = edgeLength * 2;
		roiMaxY = (int)((float)edgeLength * 0.8660254 * 2.0);
		Roi = Mat::zeros(roiMaxY, roiMaxX, CV_16U);

		Point vertice0(edgeLength / 2, 0);
		Point vertice1(edgeLength / 2 + edgeLength - 1, 0);
		Point vertice2(roiMaxX - 1, roiMaxY / 2);
		Point vertice3(edgeLength / 2 + edgeLength - 1, roiMaxY - 1);
		Point vertice4(edgeLength / 2, roiMaxY - 1);
		Point vertice5(0, roiMaxY / 2);

		line(Roi, vertice0, vertice1, 1, 1);
		line(Roi, vertice1, vertice2, 1, 1);
		line(Roi, vertice2, vertice3, 1, 1);
		line(Roi, vertice3, vertice4, 1, 1);
		line(Roi, vertice4, vertice5, 1, 1);
		line(Roi, vertice5, vertice0, 1, 1);
		unsigned short *wRoi;

		for (int y = 1; y < roiMaxY - 1; y++)
		{
			wRoi = (unsigned short *)Roi.data + roiMaxX * y;
			int x = 0;
			for (x; x < roiMaxX; x++)
			{
				if (*wRoi)
					break;
				wRoi++;
			}
			x++;
			wRoi++;
			for (x; x < roiMaxX; x++)
			{
				if (*wRoi)
					break;
				*wRoi = 1;
				wRoi++;
			}
		}

	}
	break;
	default:
		break;
	}

	int stepNr = (int)(180.0 / ProcOptions.angleStep); // angle step for computations (number of steps)

	// data vector
	float *Energy = new float[stepNr];
	float *Contrast = new float[stepNr];
	float *Correlation = new float[stepNr];
	float *Homogenity = new float[stepNr];

	float *EnergyAvg = new float[stepNr];
	float *ContrastAvg = new float[stepNr];
	float *CorrelationAvg = new float[stepNr];
	float *HomogenityAvg = new float[stepNr];

	int *Angles = new int[stepNr]; // vector for best angles histogtam
	int *AnglesCon = new int[stepNr];
	int *AnglesEne = new int[stepNr];
	int *AnglesHom = new int[stepNr];
	int *AnglesCor = new int[stepNr];

	int *AnglesAvg = new int[stepNr]; // vector for best angles histogtam
	// check how many features to compute
	float featCount = 0;
	if (ProcOptions.useContrast)
		featCount++;
	if (ProcOptions.useEnergy)
		featCount++;
	if (ProcOptions.useHomogeneity)
		featCount++;
	if (ProcOptions.useCorrelation)
		featCount++;
	//Matrix declarations
	Mat ImIn, ImInF, ImToShow, SmallIm, COM, SmallImToShow;

	steady_clock::time_point timePointOld = steady_clock::now();
	// loop through all files in the inpur directory
	for (directory_entry& FileToProcess : directory_iterator(PathToProcess))
	{

		steady_clock::time_point timePointPresent = steady_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(timePointPresent - timePointOld);
		cout << "file time: " << time_span.count() << "\n";
		timePointOld = timePointPresent;

		path InPath = FileToProcess.path();

		string OutString = ProcOptionsStr;

		// check if the filename follows the input regular expression
		if (!regex_match(InPath.filename().string().c_str(), FilePattern))
			continue;

		if (!exists(InPath))
		{
			cout << InPath.filename().string() << " File does not exist" << "\n";
			continue;
		}

		OutString += "In file - " + InPath.filename().string() + "\n";
		cout << "In file  - " << InPath.filename().string() << "\n";

		//Mat ImIn = imread(InPath.string(), CV_LOAD_IMAGE_ANYDEPTH);
		ImIn.release();
		ImIn = imread(InPath.string(), CV_LOAD_IMAGE_ANYDEPTH);
		// check if it is an image file
		if (!ImIn.size)
		{
			cout << "this is not a valid image file";
			continue;
		}

		int maxX, maxY, maxXY;
		maxX = ImIn.cols;
		maxY = ImIn.rows;
		maxXY = maxX * maxY;

		// conversion to float
		//Mat ImInF;						// ,ImInFTemp;
		ImInF.release();
		ImIn.convertTo(ImInF, CV_32F);
		// Filtering

		//
		switch (ProcOptions.preprocessType)
		{
		case 1:
			blur(ImInF, ImInF, Size(3, 3));
			break;
		case 2:
			medianBlur(ImInF, ImInF, 3);
			break;
		default:
			break;
		}


		float maxNormGlobal = ProcOptions.maxNormGlobal;
		float minNormGlobal = ProcOptions.minNormGlobal;

		switch (ProcOptions.normalisation)
		{
		case 1:
			NormParamsMinMax(ImInF, &maxNormGlobal, &minNormGlobal);
			break;
		case 2:
			NormParamsMinMax(ImInF, &maxNormGlobal, &minNormGlobal);
			break;
		case 3:
			NormParamsMeanP3Std(ImInF, &maxNormGlobal, &minNormGlobal);
			break;
		case 4:
			NormParamsMeanP3Std(ImInF, &maxNormGlobal, &minNormGlobal);
			break;
		case 5:
			NormParams1to99perc(ImInF, &maxNormGlobal, &minNormGlobal);
			break;
		case 6:
			NormParams1to99perc(ImInF, &maxNormGlobal, &minNormGlobal);
			break;
		default:
			break;
		}

		OutString += "Min Norm Global: \t" + to_string(minNormGlobal) + "\n";
		OutString += "Max Norm Global: \t" + to_string(maxNormGlobal) + "\n";

		float displayMax = ProcOptions.displayMax;
		float displayMin = ProcOptions.displayMin;

		if (!ProcOptions.useFixtDispNorm)
		{
			displayMax = maxNormGlobal;
			displayMin = minNormGlobal;
		}

		//Mat ImToShow;
		ImToShow.release();
		ImToShow = ShowImageF32PseudoColor(ImInF, displayMin, displayMax);

		if (ProcOptions.showTiles)
		{
			switch (ProcOptions.tileShape)
			{
			case 1:
				for (int y = ProcOptions.offsetTileY; y <= (maxY - ProcOptions.offsetTileY); y += ProcOptions.shiftTileY)
				{
					for (int x = ProcOptions.offsetTileX; x <= (maxX - ProcOptions.offsetTileX); x += ProcOptions.shiftTileX)
					{
						rectangle(ImToShow, Point(x - roiMaxX / 2, y - roiMaxY / 2),
							Point(x - roiMaxX / 2 + roiMaxX - 1, y - roiMaxY / 2 + roiMaxY - 1),
							Scalar(0.0, 0.0, 0.0, 0.0), ProcOptions.tileLineThickness);
					}
				}
				break;
			case 2:
				for (int y = ProcOptions.offsetTileY; y <= (maxY - ProcOptions.offsetTileY); y += ProcOptions.shiftTileY)
				{
					for (int x = ProcOptions.offsetTileX; x <= (maxX - ProcOptions.offsetTileX); x += ProcOptions.shiftTileX)
					{
						ellipse(ImToShow, Point(x, y),
							Size(roiMaxX / 2, roiMaxY / 2), 0.0, 0.0, 360.0,
							Scalar(0.0, 0.0, 0.0, 0.0), ProcOptions.tileLineThickness);
					}
				}
				break;
			case 3:
				for (int y = ProcOptions.offsetTileY; y <= (maxY - ProcOptions.offsetTileY); y += ProcOptions.shiftTileY)
				{
					for (int x = ProcOptions.offsetTileX; x <= (maxX - ProcOptions.offsetTileX); x += ProcOptions.shiftTileX)
					{
						int edgeLength = ProcOptions.maxTileX;
						Point vertice0(x - edgeLength / 2, y - (int)((float)edgeLength * 0.8660254));
						Point vertice1(x + edgeLength - edgeLength / 2, y - (int)((float)edgeLength * 0.8660254));
						Point vertice2(x + edgeLength, y);
						Point vertice3(x + edgeLength - edgeLength / 2, y + (int)((float)edgeLength * 0.8660254));
						Point vertice4(x - edgeLength / 2, y + (int)((float)edgeLength * 0.8660254));
						Point vertice5(x - edgeLength, y);

						line(ImToShow, vertice0, vertice1, Scalar(0.0, 0.0, 0.0, 0.0), ProcOptions.tileLineThickness);
						line(ImToShow, vertice1, vertice2, Scalar(0.0, 0.0, 0.0, 0.0), ProcOptions.tileLineThickness);
						line(ImToShow, vertice2, vertice3, Scalar(0.0, 0.0, 0.0, 0.0), ProcOptions.tileLineThickness);
						line(ImToShow, vertice3, vertice4, Scalar(0.0, 0.0, 0.0, 0.0), ProcOptions.tileLineThickness);
						line(ImToShow, vertice4, vertice5, Scalar(0.0, 0.0, 0.0, 0.0), ProcOptions.tileLineThickness);
						line(ImToShow, vertice5, vertice0, Scalar(0.0, 0.0, 0.0, 0.0), ProcOptions.tileLineThickness);

					}
				}
				break;
			default:
				break;
			}
		}

		if (ProcOptions.displayResult)
		{
			imshow("Image", ImToShow);
			waitKey(50);
		}

		//Mat SmallIm;
		SmallIm.release();
		int xTileNr = 0;
		int yTileNr = 0;

		string OutDataString = "";

		OutDataString += "Tile Y\tTileX\t";
		OutDataString += "Angle Contrast\tAngle Energy\tAngle Homogeneity\tAngle Correlation\tAngle\t";
		OutDataString += "Angle Contrast\tAngle Energy Avg\tAngle Homogeneity Avg\tAngle Correlation Avg\t";
		OutDataString += "Mean Intensity\tTile min norm\tTile max norm\t";
		OutDataString += "Best Angle Contrast Count\tBest Angle Energy Count\tBest Angle Homogeneity Count\tBest Angle Correlation Count\tBest Angle Count";
		OutDataString += "\n";
		for (int y = ProcOptions.offsetTileY; y <= (maxY - ProcOptions.offsetTileY); y += ProcOptions.shiftTileY)
		{
			for (int x = ProcOptions.offsetTileX; x <= (maxX - ProcOptions.offsetTileX); x += ProcOptions.shiftTileX)
			{
				ImInF(Rect(x - roiMaxX / 2, y - roiMaxY / 2, roiMaxX, roiMaxY)).copyTo(SmallIm);
				float meanSmallIm = -10.0;
				//if (ProcOptions.useMinMean)
				meanSmallIm = MatFMeanRoi(SmallIm, Roi, 1);
				bool meanCondition = 0;
				if (ProcOptions.useMinMean)
				{
					if (meanSmallIm >= ProcOptions.minMean)
						meanCondition = true;
					else
						meanCondition = false;
				}
				else
					meanCondition = true;

				float maxNorm, minNorm;

				switch (ProcOptions.normalisation)
				{
				case 1:
					NormParamsMinMax(SmallIm, Roi, 1, &maxNorm, &minNorm);
					break;
				case 2:
					maxNorm = maxNormGlobal;
					minNorm = minNormGlobal;
					break;
				case 3:
					NormParamsMeanP3Std(SmallIm, Roi, 1, &maxNorm, &minNorm);
					break;
				case 4:
					maxNorm = maxNormGlobal;
					minNorm = minNormGlobal;
					break;
				case 5:
					NormParams1to99perc(SmallIm, Roi, 1, &maxNorm, &minNorm);
					break;
				case 6:
					maxNorm = maxNormGlobal;
					minNorm = minNormGlobal;
					break;
				default:
					maxNorm = 65535.0;
					minNorm = 0.0;
					break;
				}

				for (int i = 0; i < stepNr; i++)
				{
					Angles[i] = 0;
					AnglesCon[i] = 0;
					AnglesEne[i] = 0;
					AnglesHom[i] = 0;
					AnglesCor[i] = 0;
				}

				int bestAngle;
				int maxAngle;

				int bestAngleCon;
				int maxAngleCon;

				int bestAngleEne;
				int maxAngleEne;

				int bestAngleHom;
				int maxAngleHom;

				int bestAngleCor;
				int maxAngleCor;


				int bestAngleConAvg;
				int bestAngleEneAvg;
				int bestAngleHomAvg;
				int bestAngleCorAvg;


				if (meanCondition)
				{

					// ofset loop
					for (int offset = ProcOptions.minOfset; offset <= ProcOptions.maxOfset; offset += 1)
					{
						for (int angleIndex = 0; angleIndex < stepNr; angleIndex++)
						{
							float angle = ProcOptions.angleStep * angleIndex;
							//Mat COM; // co-occurrence matrix definition
							COM.release();
							if (ProcOptions.tileShape < 2)
								COM = COMCardone4(SmallIm, offset, angle, ProcOptions.binCount, maxNorm, minNorm, ProcOptions.interpolation);
							else
								COM = COMCardoneRoi(SmallIm, Roi, offset, angle, ProcOptions.binCount, maxNorm, minNorm, ProcOptions.interpolation, 1);
							float tmpContrast, tmpEnergy, tmpHomogenity, tmpCorrelation;
							//COMParams(COM, &Contrast[angleIndex], &Energy[angleIndex], &Homogenity[angleIndex], &Correlation[angleIndex]);
							COMParams(COM, &tmpContrast, &tmpEnergy, &tmpHomogenity, &tmpCorrelation);
							Contrast[angleIndex] = tmpContrast;
							Energy[angleIndex] = tmpEnergy;
							Homogenity[angleIndex] = tmpHomogenity;
							Correlation[angleIndex] = tmpCorrelation;

							ContrastAvg[angleIndex] += tmpContrast;
							EnergyAvg[angleIndex] += tmpEnergy;
							HomogenityAvg[angleIndex] += tmpHomogenity;
							CorrelationAvg[angleIndex] = +tmpCorrelation;

						}
						// voting best angle for ofset
						int bestAngleContrast, bestAngleEnergy, bestAngleHomogenity, bestAngleCorrelation;

						bestAngleContrast = FindBestAngleMin(Contrast, stepNr);
						AnglesCon[bestAngleContrast]++;

						bestAngleEnergy = FindBestAngleMax(Energy, stepNr);
						AnglesEne[bestAngleEnergy]++;
						
						bestAngleHomogenity = FindBestAngleMax(Homogenity, stepNr);
						AnglesHom[bestAngleHomogenity]++;
						
						bestAngleCorrelation = FindBestAngleMax(Correlation, stepNr);
						AnglesCor[bestAngleCorrelation]++;
						
						// combination of features
						if (ProcOptions.useContrast)
							Angles[bestAngleContrast]++;
						if (ProcOptions.useEnergy)
							Angles[bestAngleEnergy]++;
						if (ProcOptions.useHomogeneity)
							Angles[bestAngleHomogenity]++;
						if (ProcOptions.useCorrelation)
							Angles[bestAngleCorrelation]++;
					}
					// best angle for avg
					
					bestAngleConAvg = FindBestAngleMin(Contrast, stepNr);
					bestAngleEneAvg = FindBestAngleMax(Energy, stepNr);
					bestAngleHomAvg = FindBestAngleMax(Homogenity, stepNr);
					bestAngleCorAvg = FindBestAngleMax(Correlation, stepNr);


					// look for most occurring direction
					bestAngle = 0;
					maxAngle = Angles[0];
					for (int i = 1; i < stepNr; i++)
					{
						if (maxAngle < Angles[i])
						{
							maxAngle = Angles[i];
							bestAngle = i;
						}
					}

					bestAngleCon = 0;
					maxAngleCon = AnglesCon[0];
					for (int i = 1; i < stepNr; i++)
					{
						if (maxAngleCon < AnglesCon[i])
						{
							maxAngleCon = AnglesCon[i];
							bestAngleCon = i;
						}
					}

					bestAngleEne = 0;
					maxAngleEne = AnglesEne[0];
					for (int i = 1; i < stepNr; i++)
					{
						if (maxAngleEne < AnglesEne[i])
						{
							maxAngleEne = AnglesEne[i];
							bestAngleEne = i;
						}
					}

					bestAngleHom = 0;
					maxAngleHom = AnglesHom[0];
					for (int i = 1; i < stepNr; i++)
					{
						if (maxAngleHom < AnglesHom[i])
						{
							maxAngleHom = AnglesHom[i];
							bestAngleHom = i;
						}
					}

					bestAngleCor = 0;
					maxAngleCor = AnglesCor[0];
					for (int i = 1; i < stepNr; i++)
					{
						if (maxAngleCor < AnglesCor[i])
						{
							maxAngleCor = AnglesCor[i];
							bestAngleCor = i;
						}
					}

					// show line on image

					double lineLength;
					if (ProcOptions.lineLengthPropToConfidence)
						lineLength = (double)(ProcOptions.lineHalfLength) / (ProcOptions.maxOfset - ProcOptions.minOfset + 1) / featCount * maxAngle;
					else
						lineLength = (double)(ProcOptions.lineHalfLength);
					int lineOffsetX = (int)round(lineLength *  sin((double)(bestAngle)*ProcOptions.angleStep* PI / 180.0));
					int lineOffsetY = (int)round(lineLength * cos((double)(bestAngle)*ProcOptions.angleStep* PI / 180.0));

					if (maxAngle >= ProcOptions.minHit)
					{
						//line(ImToShow, Point(barCenterX - lineOffsetX, barCenterY - lineOffsetY), Point(barCenterX + lineOffsetX, barCenterY + lineOffsetY), Scalar(0, 0.0, 0.0, 0.0), ProcOptions.imposedLineThickness);
						line(ImToShow, Point(x - lineOffsetX, y - lineOffsetY), Point(x + lineOffsetX, y + lineOffsetY), Scalar(0, 0.0, 0.0, 0.0), ProcOptions.imposedLineThickness);
					}

					if (ProcOptions.displayResult)
					{
						imshow("Image", ImToShow);
					}

					if (ProcOptions.displaySmallImage)
					{
						//Mat SmallImToShow = ShowImageF32PseudoColor(SmallIm, minNorm, maxNorm);
						SmallImToShow.release();
						SmallImToShow = ShowImageF32PseudoColor(SmallIm, minNorm, maxNorm);
						line(SmallImToShow, Point(SmallImToShow.cols / 2 - lineOffsetX, SmallImToShow.rows / 2 - lineOffsetY), Point(SmallImToShow.cols / 2 + lineOffsetX, SmallImToShow.rows / 2 + lineOffsetY), Scalar(0, 0.0, 0.0, 0.0), ProcOptions.imposedLineThickness);
						imshow("ImageSmall", ShowSolidRegionOnImage(GetContour5(Roi), SmallImToShow));
					}
				}
				
				// console output
				cout << yTileNr << "\t" << xTileNr;
				
				cout << "\t" << "ACon = ";
				if ((maxAngleCon >= ProcOptions.minHit) && meanCondition)
					cout << to_string(bestAngleCon*ProcOptions.angleStep);
				else
					cout << "NaN";
				cout << "\t";

				cout << "\t" << "AEne = ";
				if ((maxAngleEne >= ProcOptions.minHit) && meanCondition)
					cout << to_string(bestAngleEne*ProcOptions.angleStep);
				else
					cout << "NaN";
				cout << "\t";

				cout << "\t" << "AHom = ";
				if ((maxAngleHom >= ProcOptions.minHit) && meanCondition)
					cout << to_string(bestAngleHom*ProcOptions.angleStep);
				else
					cout << "NaN";
				cout << "\t";

				cout << "\t" << "ACor = ";
				if ((maxAngleCor >= ProcOptions.minHit) && meanCondition)
					cout << to_string(bestAngleCor*ProcOptions.angleStep);
				else
					cout << "NaN";
				cout << "\t";

				cout << "\t" << "A = ";
				if ((maxAngle >= ProcOptions.minHit) && meanCondition)
					cout << to_string(bestAngle*ProcOptions.angleStep);
				else
					cout << "NaN";
				cout << "\t";

				cout << "\n";

				// file output
				OutDataString += ItoStrLS(yTileNr, 2) + "\t" + ItoStrLS(xTileNr, 2) + "\t";

				if ((maxAngleCon >= ProcOptions.minHit) && meanCondition)
					OutDataString += to_string((float)bestAngleCon * ProcOptions.angleStep) + "\t";
				else
					OutDataString += "NAN\t";

				if ((maxAngleEne >= ProcOptions.minHit) && meanCondition)
					OutDataString += to_string((float)bestAngleEne * ProcOptions.angleStep) + "\t";
				else
					OutDataString += "NAN\t";

				if ((maxAngleHom >= ProcOptions.minHit) && meanCondition)
					OutDataString += to_string((float)bestAngleHom * ProcOptions.angleStep) + "\t";
				else
					OutDataString += "NAN\t";

				if ((maxAngleCor >= ProcOptions.minHit) && meanCondition)
					OutDataString += to_string((float)bestAngleCor * ProcOptions.angleStep) + "\t";
				else
					OutDataString += "NAN\t";

				if ((maxAngle >= ProcOptions.minHit) && meanCondition)
					OutDataString += to_string((float)bestAngle * ProcOptions.angleStep) + "\t";
				else
					OutDataString += "NAN\t";

				if (meanCondition)
				{
					OutDataString += to_string((float)bestAngleConAvg * ProcOptions.angleStep) + "\t";
					OutDataString += to_string((float)bestAngleEneAvg * ProcOptions.angleStep) + "\t";
					OutDataString += to_string((float)bestAngleHomAvg * ProcOptions.angleStep) + "\t";
					OutDataString += to_string((float)bestAngleCorAvg * ProcOptions.angleStep) + "\t";
				}
				else
					OutDataString += "NAN\tNAN\tNAN\tNAN\t";


				OutDataString += to_string(meanSmallIm) + "\t" + to_string(minNorm) + "\t" + to_string(maxNorm) + "\t";

				OutDataString += to_string(maxAngleCon);
				OutDataString += "\t";
				OutDataString += to_string(maxAngleEne);
				OutDataString += "\t";
				OutDataString += to_string(maxAngleHom);
				OutDataString += "\t";
				OutDataString += to_string(maxAngleCor);
				OutDataString += "\t";
				OutDataString += to_string(maxAngle); //to_string((float)(maxAngle) / (float)(ProcOptions.maxOfset - ProcOptions.minOfset + 1) / (float)featCount) + "\t";
				OutDataString += "\n";

				if (ProcOptions.displayResult || ProcOptions.displaySmallImage)
				{
					if (ProcOptions.goThru)
						waitKey(50);
					else
						waitKey(0);
				}
				xTileNr++;

			}
			yTileNr++;
			xTileNr = 0;
		}
		if (ProcOptions.imgOut)
			imwrite(ProcOptions.OutFolderName1 + InPath.filename().stem().string() + ".bmp", ImToShow);

		OutString += OutDataString;
		if (ProcOptions.textOut)
		{
			string TextFileName = ProcOptions.OutFolderName1 + InPath.filename().stem().string() + ".txt";
			std::ofstream out(TextFileName);
			out << OutString;
			out.close();
		}
	}
	//end
	//for debug only
	//string TempStr;
	//cin >> TempStr;
	return 0;
}

