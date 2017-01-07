/*
 * cdr.cpp
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#include <vector>
#include "utils.h"
#include <iostream>
#include "Argument_helper.h"
#include "Options.h"
#include "Tool.h"
#include "Document.h"
#include "genia.h"
#include "BestPerformance.h"

#include "NNgenia.h"
#include "NNgenia1.h"
#include "NNgenia2.h"
#include "NNgenia3.h"
#include "NNgenia4.h"

using namespace std;


int main(int argc, char **argv)
{

	string optionFile;
	string annotatedPath;
	string processedPath;
	string fold;



	dsr::Argument_helper ah;
	ah.new_named_string("annotated", "", "", "", annotatedPath);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("processed", "", "", "", processedPath);
	ah.new_named_string("fold", "", "", "", fold);



	ah.process(argc, argv);
	cout<<"annotated path: " <<annotatedPath <<endl;
	cout<<"processed path: "<<processedPath<<endl;



	Options options;
	options.load(optionFile);

	options.showOptions();

	Tool tool(options);

	vector< vector<Document> > groups;
	groups.resize(10); // !!!!!!!! assume it's a 10-fold cross validation
	loadAnnotatedFile(annotatedPath, groups);
	loadNlpFile(processedPath, groups);
	for(int i=0;i<groups.size();i++)
		fillTokenIdx(groups[i]);

	if(!options.embFile.empty()) {
		cout<< "load pre-trained emb"<<endl;
		tool.w2v->loadFromBinFile(options.embFile, false, true);
	}

	vector<BestPerformance> bestAll;
	int currentFold = atoi(fold.c_str());
	for(int crossValid=0;crossValid<groups.size();crossValid++) {
		if(currentFold>=0 && crossValid!=currentFold) {
			continue;
		}
		cout<<"###### group ###### "<<crossValid<<endl;

		//NNgenia nngenia(options);
		//NNgenia1 nngenia(options);
		NNgenia2 nngenia(options);
		//NNgenia3 nngenia(options);
		//NNgenia4 nngenia(options);

		// crossValid as test, crossValid+1 as dev, other as train
		vector<Document> test;
		vector<Document> dev;
		vector<Document> train;

		for(int groupIdx=0;groupIdx<groups.size();groupIdx++) {
			if(groupIdx == crossValid) {
				for(int docIdx=0; docIdx<groups[groupIdx].size(); docIdx++) {
					test.push_back(groups[groupIdx][docIdx]);
				}
			} else if(groupIdx == (crossValid+1)%groups.size()) {
				for(int docIdx=0; docIdx<groups[groupIdx].size(); docIdx++) {
					dev.push_back(groups[groupIdx][docIdx]);
				}
			} else {
				for(int docIdx=0; docIdx<groups[groupIdx].size(); docIdx++) {
					train.push_back(groups[groupIdx][docIdx]);
				}
			}

		}

		BestPerformance best = nngenia.trainAndTest(tool, test, dev, train);
		bestAll.push_back(best);
	}

	if(currentFold<0) {
		// marcro-average
		double pDev_Entity = 0;
		double rDev_Entity = 0;
		double f1Dev_Entity = 0;
		for(int i=0;i<bestAll.size();i++) {
			pDev_Entity += bestAll[i].dev_pEntity/bestAll.size();
			rDev_Entity += bestAll[i].dev_rEntity/bestAll.size();
		}
		f1Dev_Entity = f1(pDev_Entity, rDev_Entity);

		cout<<"### marcro-average ###"<<endl;
		cout<<"dev entity p "<<pDev_Entity<<endl;
		cout<<"dev entity r "<<rDev_Entity<<endl;
		cout<<"dev entity f1 "<<f1Dev_Entity<<endl;


		double pTest_Entity = 0;
		double rTest_Entity = 0;
		double f1Test_Entity = 0;
		for(int i=0;i<bestAll.size();i++) {
			pTest_Entity += bestAll[i].test_pEntity/bestAll.size();
			rTest_Entity += bestAll[i].test_rEntity/bestAll.size();
		}
		f1Test_Entity = f1(pTest_Entity, rTest_Entity);

		cout<<"test entity p "<<pTest_Entity<<endl;
		cout<<"test entity r "<<rTest_Entity<<endl;
		cout<<"test entity f1 "<<f1Test_Entity<<endl;

	}



    return 0;

}

