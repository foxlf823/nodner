
#ifndef UTILS_H_
#define UTILS_H_


#include <stdio.h>
#include <vector>
#include "Word2Vec.h"
#include "Utf.h"
#include "Entity.h"
#include "Token.h"
#include "FoxUtil.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "Document.h"
#include <list>
#include <sstream>
#include "N3L.h"
#include "Tool.h"

using namespace nr;
using namespace std;



void appendEntity(const fox::Sent& sent, int currentIdx, Entity& entity, string& labelName) {
	if(entity.spans.size()==0)
		assert(0);


	pair<int, int> & lastTkSpan = entity.tkSpans[entity.tkSpans.size()-1];

	const fox::Token& token = sent.tokens[currentIdx];

	pair<int, int> & lastSpan = entity.spans[entity.spans.size()-1];
	int whitespacetoAdd = token.begin-lastSpan.second;
	string& lastText = entity.textSpans[entity.textSpans.size()-1];

	for(int j=0;j<whitespacetoAdd;j++)
		lastText += " ";
	lastText += token.word;

	lastSpan.second = token.end;
	lastTkSpan.second = currentIdx;
	vector<string> & labelSpan = entity.labelSpans[entity.labelSpans.size()-1];
	labelSpan.push_back(labelName);

/*	if(lastTkSpan.second+1 == currentIdx) { // continuous
		pair<int, int> & lastSpan = entity.spans[entity.spans.size()-1];
		int whitespacetoAdd = token.begin-lastSpan.second;
		string& lastText = entity.textSpans[entity.textSpans.size()-1];

		for(int j=0;j<whitespacetoAdd;j++)
			lastText += " ";
		lastText += token.word;

		lastSpan.second = token.end;
		lastTkSpan.second = currentIdx;
		vector<string> & labelSpan = entity.labelSpans[entity.labelSpans.size()-1];
		labelSpan.push_back(labelName);
	} else {
		pair<int, int> span(token.begin, token.end);
		entity.spans.push_back(span);
		entity.textSpans.push_back(token.word);
		pair<int, int> tkSpan(currentIdx, currentIdx);
		entity.tkSpans.push_back(tkSpan);
		vector<string> labelSpan;
		labelSpan.push_back(labelName);
		entity.labelSpans.push_back(labelSpan);
	}*/

}

void newEntity(const fox::Sent& sent, int currentIdx, const string& labelName, Entity& entity, int entityId) {

	if(entity.spans.size()!=0)
		assert(0);

	const fox::Token& token = sent.tokens[currentIdx];

	stringstream ss;
	ss<<"T"<<entityId;
	entity.id = ss.str();
	entity.type = labelName.substr(labelName.find("_")+1);

	pair<int, int> span(token.begin, token.end);
	entity.spans.push_back(span);
	entity.textSpans.push_back(token.word);
	pair<int, int> tkSpan(currentIdx, currentIdx);
	entity.tkSpans.push_back(tkSpan);
	vector<string> labelSpan;
	labelSpan.push_back(labelName);
	entity.labelSpans.push_back(labelSpan);

}



double precision(int correct, int predict) {
	return 1.0*correct/predict;
}

double recall(int correct, int gold) {
	return 1.0*correct/gold;
}

double f1(int correct, int gold, int predict) {
	double p = precision(correct, predict);
	double r = recall(correct, gold);

	return 2*p*r/(p+r);
}

double f1(double p, double r) {

	return 2*p*r/(p+r);
}

void alphabet2vectormap(const Alphabet& alphabet, vector<string>& vector, map<string, int>& IDs) {

	for (int j = 0; j < alphabet.size(); ++j) {
		string str = alphabet.from_id(j);
		vector.push_back(str);
		IDs.insert(map<string, int>::value_type(str, j));
	}

}

template<typename T>
void array2NRMat(T * array, int sizeX, int sizeY, NRMat<T>& mat) {
	for(int i=0;i<sizeX;i++) {
		for(int j=0;j<sizeY;j++) {
			mat[i][j] = *(array+i*sizeY+j);
		}
	}
}


void initialLookupTable(LookupTable& table, PAlphabet alpha, const NRMat<dtype>& wordEmb, bool bFinetune) {

    table.elems = alpha;
    table.nVSize = wordEmb.nrows();
    table.nUNKId = table.elems->from_string(unknownkey);
    table.bFineTune = bFinetune;
    table.nDim = wordEmb.ncols();

    table.E.initial(table.nDim, table.nVSize);
    table.E.val.setZero();

	int dim1 = wordEmb.nrows();
	int dim2 = wordEmb.ncols();
	for (int idx = 0; idx < dim1; idx++) {
		for (int idy = 0; idy < dim2; idy++) {
			table.E.val(idx, idy) = wordEmb[idx][idy];
		}
	}

	if (bFinetune){
		for (int idx = 0; idx < table.nVSize; idx++){
			norm2one(table.E.val, idx);
		}
	}


  }

void stat2Alphabet(unordered_map<string, int>& stat, Alphabet& alphabet, const string& label, int wordCutOff) {

	cout << label<<" num: " << stat.size() << endl;
	alphabet.set_fixed_flag(false);
	unordered_map<string, int>::iterator feat_iter;
	for (feat_iter = stat.begin(); feat_iter != stat.end(); feat_iter++) {
		// if not fine tune, add all the words; if fine tune, add the words considering wordCutOff
		// in order to train unknown

			if (feat_iter->second > wordCutOff) {
			  alphabet.from_string(feat_iter->first);
			}

	}
	cout << "alphabet "<< label<<" num: " << alphabet.size() << endl;
	alphabet.set_fixed_flag(true);

}

void randomInitNrmat(NRMat<dtype>& nrmat, Alphabet& alphabet, int embSize, int seed, double initRange) {
	double* emb = new double[alphabet.size()*embSize];
	fox::initArray2((double *)emb, (int)alphabet.size(), embSize, 0.0);

	vector<string> known;
	map<string, int> IDs;
	alphabet2vectormap(alphabet, known, IDs);

	fox::randomInitEmb((double*)emb, embSize, known, unknownkey,
			IDs, false, initRange, seed);

	nrmat.resize(alphabet.size(), embSize);
	array2NRMat((double*) emb, alphabet.size(), embSize, nrmat);

	delete[] emb;
}

string feature_word(const fox::Token& token) {
	//string ret = token.word;
	string ret = normalize_to_lowerwithdigit(token.word);
	//string ret = normalize_to_lowerwithdigit(token.lemma);

	return ret;
}

vector<string> feature_character(const fox::Token& token) {
	vector<string> ret;
	string word = feature_word(token);
	for(int i=0;i<word.length();i++)
		ret.push_back(word.substr(i, 1));
	return ret;
}




#endif /* UTILS_H_ */
