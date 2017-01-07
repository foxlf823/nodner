
#ifndef NNgenia_H_
#define NNgenia_H_

#include <iosfwd>
#include "Options.h"
#include "Tool.h"
#include "FoxUtil.h"
#include "Utf.h"
#include "Token.h"
#include "Sent.h"
#include <sstream>
#include "N3L.h"
#include "Document.h"
#include "EnglishPos.h"
#include "Punctuation.h"
#include "Word2Vec.h"
#include "utils.h"
#include "Example.h"
#include "genia.h"
#include "BestPerformance.h"

#include "Driver.h"



using namespace nr;
using namespace std;


// schema BIO
// ignore the nested(n), overlapped(o), discontinuous(d) entities



class NNgenia {
public:
	static const int MAX_ENTITY = 3;
	string B;
	string I;
	string O;

	Options m_options;

	Driver m_driver;


	NNgenia(const Options &options):m_options(options) {
		B  = "B";
		I = "I";
		O = "O";
	}

	int NERlabelName2labelID(const string& labelName) {
		if(labelName == B) {
			return 0;
		} else if(labelName == I) {
			return 1;
		} else
			return 2;
	}

	string NERlabelID2labelName(const int labelID) {
		if(labelID == 0) {
			return B;
		} else if(labelID == 1) {
			return I;
		} else
			return O;
	}

	void getSchameLabel(fox::Token & tok, vector<Entity*> & entities, string& schemaLabel) {
		schemaLabel = O;

		// count the number that tok occurs in spans
		vector< pair<int, int> * > spansContainTok;
		vector< Entity* > entityContainSpan;
		for(int i=0;i<entities.size();i++) {
			Entity* entity = entities[i];

			for(int spanIdx=0;spanIdx<entity->spans.size();spanIdx++) {
				pair<int, int> & span = entity->spans[spanIdx];

				if(tok.begin>=span.first && tok.end<=span.second) {
					spansContainTok.push_back(&span);
					entityContainSpan.push_back(entity);
				}
			}
		}

		if(spansContainTok.size()==0)
			return ;
		else if (spansContainTok.size() > 1) { // (DB DI {HB HI) DB DI}
			for(int spanIdx=0;spanIdx<spansContainTok.size();spanIdx++) {
				pair<int, int> * span = spansContainTok[spanIdx];

				schemaLabel = O;
			}
		} else {
			pair<int, int> * currentSpan = spansContainTok[0];
			Entity * currentEntity = entityContainSpan[0];
			bool overlapped = false;

			for(int i=0;i<entities.size();i++) {
				Entity* entity = entities[i];

				for(int spanIdx=0;spanIdx<entity->spans.size();spanIdx++) {
					pair<int, int> & span = entity->spans[spanIdx];

					if(currentSpan->first == span.first && currentSpan->second == span.second)
						continue;

					if(currentSpan->second>=span.first && currentSpan->first<=span.second) {
						overlapped = true;
						break;
					}
				}
			}

			if(overlapped) { // (DB DI {HB HI) DB DI}
				schemaLabel = O;
			} else {
				if(currentEntity->spans.size() > 1) { // (DB DI)  (DB DI)
					schemaLabel = O;
				} else {
					if(tok.begin==currentSpan->first) { // (B I)
						schemaLabel = B;
					} else {
						schemaLabel = I;
					}
				}
			}




		}

		return ;
	}

	BestPerformance trainAndTest(Tool& tool,
			vector<Document> & testDocuments, vector<Document> & devDocuments, vector<Document> & trainDocuments) {
		BestPerformance ret;

		m_driver._modelparams.labelAlpha.set_fixed_flag(false);
		for(int i=0;i<MAX_ENTITY;i++) {
			string labelName = NERlabelID2labelName(i);
			m_driver._modelparams.labelAlpha.from_string(labelName);
		}
		m_driver._modelparams.labelAlpha.set_fixed_flag(true);
		cout << "total label size "<< m_driver._modelparams.labelAlpha.size() << endl;

		cout << "Creating Alphabet..." << endl;

		m_driver._modelparams.wordAlpha.clear();
		m_driver._modelparams.wordAlpha.from_string(unknownkey);
		m_driver._modelparams.wordAlpha.from_string(nullkey);

		createAlphabet(trainDocuments, tool);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devDocuments.empty())
				createAlphabet(devDocuments, tool);
			if(!testDocuments.empty())
				createAlphabet(testDocuments, tool);
		}


		NRMat<dtype> wordEmb;

		if(m_options.embFile.empty()) {
			cout<<"random emb"<<endl;

			randomInitNrmat(wordEmb, m_driver._modelparams.wordAlpha, m_options.wordEmbSize, 1000, m_options.initRange);
		} else {

			double* emb = new double[m_driver._modelparams.wordAlpha.size()*m_options.wordEmbSize];
			fox::initArray2((double *)emb, (int)m_driver._modelparams.wordAlpha.size(), m_options.wordEmbSize, 0.0);
			vector<string> known;
			map<string, int> IDs;
			alphabet2vectormap(m_driver._modelparams.wordAlpha, known, IDs);

			tool.w2v->getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

			wordEmb.resize(m_driver._modelparams.wordAlpha.size(), m_options.wordEmbSize);
			array2NRMat((double*) emb, m_driver._modelparams.wordAlpha.size(), m_options.wordEmbSize, wordEmb);

			delete[] emb;
		}


		initialLookupTable(m_driver._modelparams.words, &m_driver._modelparams.wordAlpha, wordEmb, m_options.wordEmbFineTune);



		m_driver._hyperparams.setRequared(m_options);
		m_driver.initial();



		vector<Example> trainExamples;
		initialTrainingExamples(tool, trainDocuments, trainExamples);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;


		dtype bestDIS = 0;
		int inputSize = trainExamples.size();
		int batchBlock = inputSize / m_options.batchSize;
		if (inputSize % m_options.batchSize != 0)
			batchBlock++;

		std::vector<int> indexes;
		for (int i = 0; i < inputSize; ++i)
			indexes.push_back(i);

		static Metric eval;
		static vector<Example> subExamples;


		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

			cout << "##### Iteration " << iter << std::endl;

		    eval.reset();


		    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
				subExamples.clear();
				int start_pos = updateIter * m_options.batchSize;
				int end_pos = (updateIter + 1) * m_options.batchSize;
				if (end_pos > inputSize)
					end_pos = inputSize;

				for (int idy = start_pos; idy < end_pos; idy++) {
					subExamples.push_back(trainExamples[indexes[idy]]);
				}

				int curUpdateIter = iter * batchBlock + updateIter;
				dtype cost = m_driver.train(subExamples, curUpdateIter);
				//m_driver.checkgrad(subExamples, curUpdateIter + 1);

				eval.overall_label_count += m_driver._eval.overall_label_count;
				eval.correct_label_count += m_driver._eval.correct_label_count;

/*				if ((curUpdateIter + 1) % m_options.verboseIter == 0) {

					std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
					std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
				}*/
				m_driver.updateModel();
		    }

		    // an iteration end, begin to evaluate
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	BestPerformance currentDev = evaluateOnDev(tool, devDocuments);


				if (currentDev.dev_f1Entity > bestDIS) {
					cout << "Exceeds best performance of " << bestDIS << endl;
					bestDIS = currentDev.dev_f1Entity;
					ret.dev_pEntity = currentDev.dev_pEntity;
					ret.dev_rEntity = currentDev.dev_rEntity;
					ret.dev_f1Entity = currentDev.dev_f1Entity;

					if(testDocuments.size()>0) {

						BestPerformance currentTest = test(tool, testDocuments);
						ret.test_pEntity = currentTest.test_pEntity;
						ret.test_rEntity = currentTest.test_rEntity;
						ret.test_f1Entity = currentTest.test_f1Entity;

					}

				}



		    } // devExamples > 0

		} // for iter

		return ret;

	}


	void initialTrainingExamples(Tool& tool, vector<Document>& documents, vector<Example>& examples) {

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			Document& doc = documents[docIdx];


			for(int sentIdx=0;sentIdx<doc.sentences.size();sentIdx++) {

				Sentence & sent = doc.sentences[sentIdx];

				vector<Entity*> entities;
				findEntityInSent(sent, entities);

				Example eg;
				generateOneNerExample(eg, sent.info, entities, false);

				examples.push_back(eg);


			} // sent


		} // doc

	}

	BestPerformance evaluateOnDev(Tool& tool, vector<Document>& documents) {

		BestPerformance ret;

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			Document& doc = documents[docIdx];

			for(int sentIdx=0;sentIdx<doc.sentences.size();sentIdx++) {

				Sentence & sent = doc.sentences[sentIdx];

				vector<Entity*> entities;
				findEntityInSent(sent, entities);

				Example eg;
				generateOneNerExample(eg, sent.info, entities, true);

				vector<int> labelIdx;
				m_driver.predict(eg.m_features, labelIdx);

				vector<Entity> predictEntitiesInThisSent;
				decode(sent.info, labelIdx, predictEntitiesInThisSent);


				// evaluate by ourselves
				ctGoldEntity += entities.size();
				ctPredictEntity += predictEntitiesInThisSent.size();
				for(int i=0;i<predictEntitiesInThisSent.size();i++) {

					int j=0;
					for(;j<entities.size();j++) {
						if(predictEntitiesInThisSent[i].equalsBoundary(*entities[j])) {
							ctCorrectEntity ++;
							break;

						}
					}

				}


			} // sent




		} // doc

		ret.dev_pEntity = precision(ctCorrectEntity, ctPredictEntity);
		ret.dev_rEntity = recall(ctCorrectEntity, ctGoldEntity);
		ret.dev_f1Entity = f1(ctCorrectEntity, ctGoldEntity, ctPredictEntity);

		cout<<"dev entity p: "<<ret.dev_pEntity<<", r:"<<ret.dev_rEntity<<", f1:"<<ret.dev_f1Entity<<endl;

		return ret;


	}

	BestPerformance test(Tool& tool, vector<Document>& documents) {
		BestPerformance ret;

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			Document& doc = documents[docIdx];

			for(int sentIdx=0;sentIdx<doc.sentences.size();sentIdx++) {

				Sentence & sent = doc.sentences[sentIdx];

				vector<Entity*> entities;
				findEntityInSent(sent, entities);

				Example eg;
				generateOneNerExample(eg, sent.info, entities, true);

				vector<int> labelIdx;
				m_driver.predict(eg.m_features, labelIdx);


				vector<Entity> predictEntitiesInThisSent;
				decode(sent.info, labelIdx, predictEntitiesInThisSent);


				// evaluate by ourselves
				ctGoldEntity += entities.size();
				ctPredictEntity += predictEntitiesInThisSent.size();
				for(int i=0;i<predictEntitiesInThisSent.size();i++) {

					int j=0;
					for(;j<entities.size();j++) {
						if(predictEntitiesInThisSent[i].equalsBoundary(*entities[j])) {
							ctCorrectEntity ++;
							break;

						}
					}

				}



			} // sent


		} // doc

		ret.test_pEntity = precision(ctCorrectEntity, ctPredictEntity);
		ret.test_rEntity = recall(ctCorrectEntity, ctGoldEntity);
		ret.test_f1Entity = f1(ctCorrectEntity, ctGoldEntity, ctPredictEntity);

		cout<<"test entity p: "<<ret.test_pEntity<<", r:"<<ret.test_rEntity<<", f1:"<<ret.test_f1Entity<<endl;

		return ret;

	}


	void decode(fox::Sent& sent, vector<int>& labelIdx, vector<Entity> & anwserEntities) {

		int seq_size = labelIdx.size();
		vector<string> outputs;
		outputs.resize(seq_size);
		for (int idx = 0; idx < seq_size; idx++) {
			outputs[idx] = m_driver._modelparams.labelAlpha.from_id(labelIdx[idx], O);
		}

		for(int idx=0;idx<sent.tokens.size();idx++) {
			string& labelName = outputs[idx];
			fox::Token& token = sent.tokens[idx];

			// decode entity label
			if(labelName == B) {
				Entity entity;
				newEntity(sent, idx, labelName, entity, 0);
				anwserEntities.push_back(entity);
			} else if(labelName == I) {
				if(checkWrongState(outputs, idx+1)) {
					Entity& entity = anwserEntities[anwserEntities.size()-1];
					appendEntity(sent, idx, entity, labelName);
				}
			}

		} // token
	}

	// Only used when current label is I or L, check state from back to front
	// in case that "O I I", etc.
	bool checkWrongState(vector<string>& labelSequence, int size) {
		int positionNew = -1; // the latest type-consistent B
		int positionOther = -1; // other label except type-consistent I

		string& currentLabel = labelSequence[size-1];
		if(currentLabel!=I)
			assert(0);

		for(int j=size-2;j>=0;j--) {

			if(positionNew==-1 && labelSequence[j]==B) {
				positionNew = j;
			} else if(positionOther==-1 && labelSequence[j]!=I) {
				positionOther = j;
			}

			if(positionOther!=-1 && positionNew!=-1)
				break;
		}

		if(positionNew == -1)
			return false;
		else if(positionOther<positionNew)
			return true;
		else
			return false;
	}

	void generateOneNerExample(Example& eg, fox::Sent& sent, vector<Entity*> & entities, bool bPredict) {


		if(!bPredict) {

			for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
				fox::Token& token = sent.tokens[tokenIdx];

				string labelName = O;

				getSchameLabel(token, entities, labelName);

				int labelID = NERlabelName2labelID(labelName);
				vector<dtype> labelsForThisToken;
				for(int i=0;i<MAX_ENTITY;i++)
					labelsForThisToken.push_back(0.0);
				labelsForThisToken[labelID] = 1.0;

				eg.m_labels.push_back(labelsForThisToken);

			} // token
		}

		for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
			fox::Token& token = sent.tokens[tokenIdx];

			Feature featureForThisToken;
			featureForThisToken.words.push_back(feature_word(token));

			eg.m_features.push_back(featureForThisToken);

		} // token


	}


	void createAlphabet (vector<Document>& documents, Tool& tool) {


		unordered_map<string, int> word_stat;


		for(int docIdx=0;docIdx<documents.size();docIdx++) {

			for(int i=0;i<documents[docIdx].sentences.size();i++) {

				for(int j=0;j<documents[docIdx].sentences[i].info.tokens.size();j++) {

					string curword = feature_word(documents[docIdx].sentences[i].info.tokens[j]);
					word_stat[curword]++;


				}


			}


		}

		stat2Alphabet(word_stat, m_driver._modelparams.wordAlpha, "word", m_options.wordCutOff);



	}









};



#endif /* NNBB3_H_ */

