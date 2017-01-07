
#ifndef NNclef2_H_
#define NNclef2_H_

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
#include "clef2013.h"

#include "Driver.h"



using namespace nr;
using namespace std;


// schema BIOHD1234



class NNclef2 {
public:
	static const int MAX_ENTITY = 13;
	string B;
	string I;
	string HB;
	string HI;
	string D1B;
	string D1I;
	string D2B;
	string D2I;
	string D3B;
	string D3I;
	string D4B;
	string D4I;
	string O;


	Options m_options;

	Driver m_driver;


	NNclef2(const Options &options):m_options(options) {
		B  = "B";
		I = "I";
		HB = "HB";
		HI = "HI";
		D1B = "D1B";
		D1I = "D1I";
		D2B = "D2B";
		D2I = "D2I";
		D3B = "D3B";
		D3I = "D3I";
		D4B = "D4B";
		D4I = "D4I";
		O = "O";

	}

	int NERlabelName2labelID(const string& labelName) {
		if(labelName == B) {
			return 0;
		} else if(labelName == I) {
			return 1;
		} else if(labelName == HB) {
			return 2;
		} else if(labelName == HI) {
			return 3;
		} else if(labelName == D1B) {
			return 4;
		} else if(labelName == D1I) {
			return 5;
		} else if(labelName == D2B) {
			return 6;
		} else if(labelName == D2I) {
			return 7;
		} else if(labelName == D3B) {
			return 8;
		} else if(labelName == D3I) {
			return 9;
		} else if(labelName == D4B) {
			return 10;
		} else if(labelName == D4I) {
			return 11;
		}

		else
			return 12;
	}

	string NERlabelID2labelName(const int labelID) {
		if(labelID == 0) {
			return B;
		} else if(labelID == 1) {
			return I;
		} else if(labelID == 2) {
			return HB;
		} else if(labelID == 3) {
			return HI;
		} else if(labelID == 4) {
			return D1B;
		} else if(labelID == 5) {
			return D1I;
		} else if(labelID == 6) {
			return D2B;
		} else if(labelID == 7) {
			return D2I;
		}  else if(labelID == 8) {
			return D3B;
		} else if(labelID == 9) {
			return D3I;
		}  else if(labelID == 10) {
			return D4B;
		} else if(labelID == 11) {
			return D4I;
		}

		else
			return O;
	}


	void getSchameLabel(fox::Sent& sent, int tokenIdx, vector<Entity*> & entities, string& schemaLabel) {
		schemaLabel = O;
		fox::Token& tok = sent.tokens[tokenIdx];

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

				if(tok.begin==span->first) {
					schemaLabel = HB;
					break;
				} else {
					schemaLabel = HI;
				}
			}
		} else {
			pair<int, int> * currentSpan = spansContainTok[0];
			Entity * currentEntity = entityContainSpan[0];
			bool overlapped = false;
			pair<int, int> * overlappedSpan = NULL;

			for(int i=0;i<entities.size();i++) {
				Entity* entity = entities[i];

				for(int spanIdx=0;spanIdx<entity->spans.size();spanIdx++) {
					pair<int, int> & span = entity->spans[spanIdx];

					if(currentSpan->first == span.first && currentSpan->second == span.second)
						continue;

					if(currentSpan->second>=span.first && currentSpan->first<=span.second) {
						overlapped = true;
						overlappedSpan = &span;
						break;
					}
				}
			}

			if(overlapped) { // (DB DI {HB HI) DB DI}
				int i=0;
				for(;i<sent.tokens.size();i++) {
					if(sent.tokens[i].begin >= overlappedSpan->second)
						break;
				}

				if(tok.begin==currentSpan->first) {
					schemaLabel = D3B;
				} else if(i == tokenIdx) {
					schemaLabel = D1B;
				} else {
					if(sent.tokens[i].begin < overlappedSpan->first)
						schemaLabel = D3I;
					else
						schemaLabel = D1I;
				}
			} else {
				if(currentEntity->spans.size() > 1) { // (DB DI)  (DB DI)

					pair<int, int> * otherSpan = NULL;
					for(int spanIdx=0;spanIdx<currentEntity->spans.size();spanIdx++) {
						pair<int, int> & span = currentEntity->spans[spanIdx];
						if(currentSpan->first == span.first && currentSpan->second == span.second)
							continue;

						otherSpan = &span; // assume only one other discontinuous span
						break;
					}

					int coutOtherSpan = 0;
					for(int i=0;i<entities.size();i++) {
						Entity* entity = entities[i];

						for(int spanIdx=0;spanIdx<entity->spans.size();spanIdx++) {
							pair<int, int> & span = entity->spans[spanIdx];

							if(otherSpan->first == span.first && otherSpan->second == span.second) {
								coutOtherSpan++;
							}
						}
					}

					if(coutOtherSpan>1) {
						if(currentSpan->first < otherSpan->first) {
							if(tok.begin==currentSpan->first) {
								schemaLabel = D3B;
							} else {
								schemaLabel = D3I;
							}
						} else {
							if(tok.begin==currentSpan->first) {
								schemaLabel = D1B;
							} else {
								schemaLabel = D1I;
							}
						}

					} else {
						if(currentSpan->first < otherSpan->first) {
							if(tok.begin==currentSpan->first) {
								schemaLabel = D4B;
							} else {
								schemaLabel = D4I;
							}
						} else {
							if(tok.begin==currentSpan->first) {
								schemaLabel = D2B;
							} else {
								schemaLabel = D2I;
							}
						}
					}


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


	void trainAndTest(const string& trainFile, const string& devFile, const string& testFile,
			Tool& tool,
			const string& trainNlpFile, const string& devNlpFile, const string& testNlpFile) {


		// load train data
		vector<Document> trainDocuments;
		loadAnnotatedFile(trainFile, trainDocuments);
		loadNlpFile(trainNlpFile, trainDocuments);
		fillTokenIdx(trainDocuments);

		vector<Document> devDocuments;
		if(!devFile.empty()) {
			loadAnnotatedFile(devFile, devDocuments);
			loadNlpFile(devNlpFile, devDocuments);
			fillTokenIdx(devDocuments);
		}
		vector<Document> testDocuments;
		if(!testFile.empty()) {
			loadAnnotatedFile(testFile, testDocuments);
			loadNlpFile(testNlpFile, testDocuments);
			fillTokenIdx(testDocuments);
		}

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
			cout<< "load pre-trained emb"<<endl;
			tool.w2v->loadFromBinFile(m_options.embFile, false, true);

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

		static Metric eval, metric_dev;
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

		    	evaluateOnDev(tool, devDocuments, metric_dev);

				if (metric_dev.getAccuracy() > bestDIS) {
					cout << "Exceeds best performance of " << bestDIS << endl;
					bestDIS = metric_dev.getAccuracy();

/*					if(testDocuments.size()>0) {

						// clear outpu dir
						string s = "rm -f "+m_options.output+"/*";
						system(s.c_str());

						test(tool, testDocuments, metric_dev);

					}*/

				}



		    } // devExamples > 0

		} // for iter



	}


	void initialTrainingExamples(Tool& tool, vector<Document>& documents, vector<Example>& examples) {

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			Document& doc = documents[docIdx];


			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				if(sentIdx==0)
					continue; // the first row has no mentions

				fox::Sent & sent = doc.sents[sentIdx];

				if(sent.tokens.size()==1 && sent.tokens[0].word=="$")
					continue; // the empty line

				vector<Entity*> entities;
				findEntityInSent(sent.begin, sent.end, doc, entities);

				Example eg;
				generateOneNerExample(eg, sent, entities, false);

				examples.push_back(eg);


			} // sent


		} // doc

	}

	void evaluateOnDev(Tool& tool, vector<Document>& documents, Metric& metric_dev) {
    	metric_dev.reset();

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			Document& doc = documents[docIdx];

			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				if(sentIdx==0)
					continue; // the first row has no mentions

				fox::Sent & sent = doc.sents[sentIdx];

				if(sent.tokens.size()==1 && sent.tokens[0].word=="$")
					continue; // the empty line

				vector<Entity*> entities;
				findEntityInSent(sent.begin, sent.end, doc, entities);

				Example eg;
				generateOneNerExample(eg, sent, entities, true);

				vector<int> labelIdx;
				m_driver.predict(eg.m_features, labelIdx);


				vector<Entity> predictEntitiesInThisSent;
				decode(sent, labelIdx, predictEntitiesInThisSent);


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

		metric_dev.overall_label_count = ctGoldEntity;
		metric_dev.predicated_label_count = ctPredictEntity;
		metric_dev.correct_label_count = ctCorrectEntity;
		metric_dev.print();

	}

	void test(Tool& tool, vector<Document>& documents, Metric& metric_dev) {
		metric_dev.reset();

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			Document& doc = documents[docIdx];
			vector<Entity> anwserEntities;

			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				if(sentIdx==0)
					continue; // the first row has no mentions

				fox::Sent & sent = doc.sents[sentIdx];

				if(sent.tokens.size()==1 && sent.tokens[0].word=="$")
					continue; // the empty line

				vector<Entity*> entities;
				findEntityInSent(sent.begin, sent.end, doc, entities);

				Example eg;
				generateOneNerExample(eg, sent, entities, true);

				vector<int> labelIdx;
				m_driver.predict(eg.m_features, labelIdx);

				vector<Entity> predictEntitiesInThisSent;
				decode(sent, labelIdx, predictEntitiesInThisSent);


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


					anwserEntities.push_back(predictEntitiesInThisSent[i]);
				}

			} // sent


			outputResults(doc, anwserEntities, m_options.output);

		} // doc

		metric_dev.overall_label_count = ctGoldEntity;
		metric_dev.predicated_label_count = ctPredictEntity;
		metric_dev.correct_label_count = ctCorrectEntity;
		metric_dev.print();

	}


	void decode(fox::Sent& sent, vector<int>& labelIdx, vector<Entity> & anwserEntities) {
		vector<Entity> entities;

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
			if(labelName == B || labelName == HB || labelName == D1B || labelName == D2B
					|| labelName == D3B || labelName == D4B) {
				Entity entity;
				newEntity(sent, idx, labelName, entity, 0);
				entities.push_back(entity);
			} else if(labelName == I || labelName == HI || labelName == D1I || labelName == D2I
					|| labelName == D3I || labelName == D4I) {
				if(checkWrongState(outputs, idx+1)) {
					Entity& entity = entities[entities.size()-1];
					appendEntity(sent, idx, entity, labelName);
				}
			}

		} // token

		// post-processing to rebuild entities
		vector<Entity> postEntities;

		vector<Entity*> HB_HI;
		vector<Entity*> D1B_D1I;
		vector<Entity*> D2B_D2I;
		vector<Entity*> D3B_D3I;
		vector<Entity*> D4B_D4I;


		for(int entityIdx=0;entityIdx<entities.size();entityIdx++) {
			Entity & temp = entities[entityIdx];

			vector<string> & labelSpan = temp.labelSpans[0];

			if(labelSpan[0]==HB) {
				HB_HI.push_back(&temp);
			} else if(labelSpan[0]==D1B) {
				D1B_D1I.push_back(&temp);
			} else if(labelSpan[0]==D2B) {
				D2B_D2I.push_back(&temp);
			} else if(labelSpan[0]==D3B) {
				D3B_D3I.push_back(&temp);
			} else if(labelSpan[0]==D4B) {
				D4B_D4I.push_back(&temp);
			}

			else {
				postEntities.push_back(temp);
			}
		}

		if(HB_HI.size()!=0) {
			for(int idxD1B=0;idxD1B<D1B_D1I.size();idxD1B++) {
				Entity & d1b = *D1B_D1I[idxD1B];

				// combine with the nearest head entity at left
				Entity * target = NULL;
				for(int idxHB=0;idxHB<HB_HI.size();idxHB++) {
					Entity & hb = *HB_HI[idxHB];

					if(hb.spans[0].first<d1b.spans[0].first) {
						target = & hb;
					} else {
						break;
					}
				}
				if(target == NULL) {
					// error prediction
				} else {
					Entity combined;
					combineTwoEntity(d1b, *target, combined);
					postEntities.push_back(combined);
					if(D1B_D1I.size()==1)
						postEntities.push_back(*target);
				}
			}

			for(int idxD3B=0;idxD3B<D3B_D3I.size();idxD3B++) {
				Entity & d3b = *D3B_D3I[idxD3B];

				// combine with the nearest head entity at right
				Entity * target = NULL;
				for(int idxHB=HB_HI.size()-1;idxHB>=0;idxHB--) {
					Entity & hb = *HB_HI[idxHB];

					if(hb.spans[0].first>d3b.spans[0].first) {
						target = & hb;
					} else {
						break;
					}
				}
				if(target == NULL) {
					// error prediction
				} else {
					Entity combined;
					combineTwoEntity(d3b, *target, combined);
					postEntities.push_back(combined);
					if(D3B_D3I.size()==1)
						postEntities.push_back(*target);
				}
			}
		} else {
			for(int idxD2B=0;idxD2B<D2B_D2I.size();idxD2B++) {
				Entity & d2b = *D2B_D2I[idxD2B];

				// combine with the nearest non-head entity at left
				Entity * target = NULL;
				for(int idxDB=0;idxDB<D1B_D1I.size();idxDB++) {
					Entity & db = *D1B_D1I[idxDB];

					if(db.spans[0].first<d2b.spans[0].first) {
						target = & db;
					} else {
						break;
					}
				}

				for(int idxDB=0;idxDB<D2B_D2I.size();idxDB++) {
					Entity & db = *D2B_D2I[idxDB];

					if(db.spans[0].first<d2b.spans[0].first) {
						if(target!=NULL && target->spans[0].first<db.spans[0].first)
							target = & db;
						else
							target = & db;
					} else {
						break;
					}
				}

				for(int idxDB=0;idxDB<D3B_D3I.size();idxDB++) {
					Entity & db = *D3B_D3I[idxDB];

					if(db.spans[0].first<d2b.spans[0].first) {
						if(target!=NULL && target->spans[0].first<db.spans[0].first)
							target = & db;
						else
							target = & db;
					} else {
						break;
					}
				}

				for(int idxDB=0;idxDB<D4B_D4I.size();idxDB++) {
					Entity & db = *D4B_D4I[idxDB];

					if(db.spans[0].first<d2b.spans[0].first) {
						if(target!=NULL && target->spans[0].first<db.spans[0].first)
							target = & db;
						else
							target = & db;
					} else {
						break;
					}
				}

				if(target == NULL) {
					// error prediction
				} else {
					Entity combined;
					combineTwoEntity(d2b, *target, combined);
					postEntities.push_back(combined);
				}
			}

			for(int idxD4B=0;idxD4B<D4B_D4I.size();idxD4B++) {
				Entity & d4b = *D4B_D4I[idxD4B];

				// combine with the nearest non-head entity at right
				Entity * target = NULL;
				for(int idxDB=D1B_D1I.size()-1;idxDB>=0;idxDB--) {
					Entity & db = *D1B_D1I[idxDB];

					if(db.spans[0].first>d4b.spans[0].first) {
						target = & db;
					} else {
						break;
					}
				}

				for(int idxDB=D2B_D2I.size()-1;idxDB>=0;idxDB--) {
					Entity & db = *D2B_D2I[idxDB];

					if(db.spans[0].first>d4b.spans[0].first) {
						if(target!=NULL && target->spans[0].first>db.spans[0].first)
							target = & db;
						else
							target = & db;
					} else {
						break;
					}
				}

				for(int idxDB=D3B_D3I.size()-1;idxDB>=0;idxDB--) {
					Entity & db = *D3B_D3I[idxDB];

					if(db.spans[0].first>d4b.spans[0].first) {
						if(target!=NULL && target->spans[0].first>db.spans[0].first)
							target = & db;
						else
							target = & db;
					} else {
						break;
					}
				}

				for(int idxDB=D4B_D4I.size()-1;idxDB>=0;idxDB--) {
					Entity & db = *D4B_D4I[idxDB];

					if(db.spans[0].first>d4b.spans[0].first) {
						if(target!=NULL && target->spans[0].first>db.spans[0].first)
							target = & db;
						else
							target = & db;
					} else {
						break;
					}
				}

				if(target == NULL) {
					// error prediction
				} else {
					Entity combined;
					combineTwoEntity(d4b, *target, combined);
					postEntities.push_back(combined);
				}
			}
		}


		// resort by start position and remove the same entity
		for(int entityIdx=0;entityIdx<postEntities.size();entityIdx++) {
			Entity & temp = postEntities[entityIdx];

			bool isIn = false;
			for(int i=0;i<anwserEntities.size();i++) {
				Entity & anwser = anwserEntities[i];
				if(anwser.equalsBoundary(temp)) {
					isIn = true;
					break;
				}
			}

			if(isIn == false) {

				vector<Entity>::iterator iter = anwserEntities.begin();
				for(;iter!=anwserEntities.end();iter++) {
					if(iter->spans[0].first>temp.spans[0].first) {
						break;
					}
				}

				anwserEntities.insert(iter, temp);
			}


		}
	}

	// Only used when current label is I or L, check state from back to front
	// in case that "O I I", etc.
	bool checkWrongState(vector<string>& labelSequence, int size) {
		int positionNew = -1; // the latest type-consistent B
		int positionOther = -1; // other label except type-consistent I

		string& currentLabel = labelSequence[size-1];
		if(currentLabel!=I && currentLabel!=HI && currentLabel!=D1I && currentLabel!=D2I
				&& currentLabel!=D3I && currentLabel!=D4I)
			assert(0);

		for(int j=size-2;j>=0;j--) {
			if(currentLabel==I) {
				if(positionNew==-1 && labelSequence[j]==B) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I) {
					positionOther = j;
				}
			} else if(currentLabel==HI) {
				if(positionNew==-1 && labelSequence[j]==HB) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=HI) {
					positionOther = j;
				}
			} else if(currentLabel==D1I) {
				if(positionNew==-1 && labelSequence[j]==D1B) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=D1I) {
					positionOther = j;
				}
			} else if(currentLabel==D2I) {
				if(positionNew==-1 && labelSequence[j]==D2B) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=D2I) {
					positionOther = j;
				}
			} else if(currentLabel==D3I) {
				if(positionNew==-1 && labelSequence[j]==D3B) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=D3I) {
					positionOther = j;
				}
			} else {
				if(positionNew==-1 && labelSequence[j]==D4B) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=D4I) {
					positionOther = j;
				}
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

				getSchameLabel(sent, tokenIdx, entities, labelName);

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

			for(int i=0;i<documents[docIdx].sents.size();i++) {

				for(int j=0;j<documents[docIdx].sents[i].tokens.size();j++) {

					string curword = feature_word(documents[docIdx].sents[i].tokens[j]);
					word_stat[curword]++;


				}


			}


		}

		stat2Alphabet(word_stat, m_driver._modelparams.wordAlpha, "word", m_options.wordCutOff);


	}









};



#endif /* NNBB3_H_ */

