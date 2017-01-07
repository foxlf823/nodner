
#ifndef NNclef3_H_
#define NNclef3_H_

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
#include "Dependency.h"

#include "Driver.h"
#include "Driver_relation.h"



using namespace nr;
using namespace std;


// schema BIO
// combine BIO automatically


class NNclef3 {
public:
	static const int MAX_ENTITY = 7;
	string B;
	string I;
	string HB;
	string HI;
	string DB;
	string DI;
	string O;

	static const int MAX_RELATION = 2;
	string N;
	string Y;


	Options m_options;

	Driver m_driver;

	Driver_relation m_driver_relation;


	NNclef3(const Options &options):m_options(options) {
		B  = "B";
		I = "I";
		HB = "HB";
		HI = "HI";
		DB = "DB";
		DI = "DI";
		O = "O";

		N = "N";
		Y = "Y";
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
		} else if(labelName == DB) {
			return 4;
		} else if(labelName == DI) {
			return 5;
		}

		else
			return 6;
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
			return DB;
		} else if(labelID == 5) {
			return DI;
		}

		else
			return O;
	}

	int RellabelName2labelID(const string& labelName) {

		if(labelName == N) {
			return 0;
		} else
			return 1;

	}

	string RellabelID2labelName(const int labelID) {

		if(labelID == 0) {
			return N;
		} else
			return Y;

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
					schemaLabel = DB;
				} else if(i == tokenIdx) {
					schemaLabel = DB;
				} else {
					schemaLabel = DI;
				}
			} else {
				if(currentEntity->spans.size() > 1) { // (DB DI)  (DB DI)
					if(tok.begin==currentSpan->first) {
						schemaLabel = DB;
					} else {
						schemaLabel = DI;
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

		m_driver_relation._modelparams.labelAlpha_relation.set_fixed_flag(false);
		m_driver_relation._modelparams.labelAlpha_relation.from_string(N);
		m_driver_relation._modelparams.labelAlpha_relation.from_string(Y);
		m_driver_relation._modelparams.labelAlpha_relation.set_fixed_flag(true);
		cout << "total relation label size "<< m_driver_relation._modelparams.labelAlpha_relation.size() << endl;

		cout << "Creating Alphabet..." << endl;

		m_driver._modelparams.wordAlpha.clear();
		m_driver._modelparams.wordAlpha.from_string(unknownkey);
		m_driver._modelparams.wordAlpha.from_string(nullkey);

		m_driver_relation._modelparams.wordAlpha.clear();
		m_driver_relation._modelparams.wordAlpha.from_string(unknownkey);
		m_driver_relation._modelparams.wordAlpha.from_string(nullkey);


		createAlphabet(trainDocuments, tool);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devDocuments.empty())
				createAlphabet(devDocuments, tool);
			if(!testDocuments.empty())
				createAlphabet(testDocuments, tool);
		}


		// here we use one copy to initialize the wordAlpha of both m_driver and m_driver_relation
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

		initialLookupTable(m_driver_relation._modelparams.words, &m_driver_relation._modelparams.wordAlpha, wordEmb, m_options.wordEmbFineTune);


		m_driver._hyperparams.setRequared(m_options);
		m_driver.initial();

		m_driver_relation._hyperparams.setRequared(m_options);
		m_driver_relation.initial();



		vector<Example> trainExamples;
		initialTrainingExamples(tool, trainDocuments, trainExamples);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;

		vector<Example> trainRelationExamples;
		initialTrainingRelationExamples(tool, trainDocuments, trainRelationExamples);
		cout<<"Total train relation example number: "<<trainRelationExamples.size()<<endl;


		dtype bestDIS = 0;
		int inputSize = trainExamples.size()+trainRelationExamples.size();
		int batchBlock = inputSize / m_options.batchSize;
		if (inputSize % m_options.batchSize != 0)
			batchBlock++;

		std::vector<int> indexes;
		for (int i = 0; i < inputSize; ++i)
			indexes.push_back(i);

		static Metric eval, metric_dev;
		static vector<Example> subExamples;
		static vector<Example> subRelationExamples;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

			cout << "##### Iteration " << iter << std::endl;

		    eval.reset();


		    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
				subExamples.clear();
				subRelationExamples.clear();
				int start_pos = updateIter * m_options.batchSize;
				int end_pos = (updateIter + 1) * m_options.batchSize;
				if (end_pos > inputSize)
					end_pos = inputSize;


				for (int idy = start_pos; idy < end_pos; idy++) {
					if(idy>=0 && idy<trainExamples.size())
						subExamples.push_back(trainExamples[indexes[idy]]);
					else
						subRelationExamples.push_back(trainRelationExamples[indexes[idy-trainExamples.size()]]);
				}

				int curUpdateIter = iter * batchBlock + updateIter;

				if(subExamples.size()>0) {
					dtype cost = m_driver.train(subExamples, curUpdateIter);
					//m_driver.checkgrad(subExamples, curUpdateIter + 1);
				}

				if(subRelationExamples.size()>0) {
					m_driver_relation.train(subRelationExamples, curUpdateIter);
					// m_driver_relation.checkgrad(subExamples, curUpdateIter + 1);
				}


/*				eval.overall_label_count += m_driver._eval.overall_label_count;
				eval.correct_label_count += m_driver._eval.correct_label_count;

				if ((curUpdateIter + 1) % m_options.verboseIter == 0) {

					std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
					std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
				}*/
				if(subExamples.size()>0) {
					m_driver.updateModel();
				}

				if(subRelationExamples.size()>0) {
					m_driver_relation.updateModel();
				}


		    }

		    // an iteration end, begin to evaluate
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	evaluateOnDev(tool, devDocuments, metric_dev);

				if (metric_dev.getAccuracy() > bestDIS) {
					cout << "Exceeds best performance of " << bestDIS << endl;
					bestDIS = metric_dev.getAccuracy();

/*					if(testDocuments.size()>0) {

						// clear output dir
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

	void initialTrainingRelationExamples(Tool& tool, vector<Document>& documents, vector<Example>& examples) {

		int countPositive = 0;
		int countNegative = 0;

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

				// generate spans based on the gold answers
				vector<string> outputs;
				for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
					fox::Token& token = sent.tokens[tokenIdx];

					string labelName = O;
					getSchameLabel(sent, tokenIdx, entities, labelName);
					outputs.push_back(labelName);
				}

				vector<Entity> spanEntities;
				decode(sent, outputs, spanEntities);


				// for each pair of spans (HB HI) or (DB DI), determine their relation by entities
				for(int latterIdx=0;latterIdx<spanEntities.size();latterIdx++) {
					Entity* latterSpan = &spanEntities[latterIdx];

					if(latterSpan->labelSpans[0][0] != HB && latterSpan->labelSpans[0][0] != DB)
						continue;

					// find entities that contain latter span
					vector< Entity* > entityContainLatterSpan;
					for(int entityIdx=0;entityIdx<entities.size();entityIdx++) {
						Entity* entity = entities[entityIdx];

						for(int spanIdx=0;spanIdx<entity->spans.size();spanIdx++) {
							pair<int, int> & span = entity->spans[spanIdx];

							if(span.first<=latterSpan->spans[0].first && span.second>=latterSpan->spans[0].second) {
								entityContainLatterSpan.push_back(entity);
								break;
							}
						}
					}


					for(int formerIdx=0;formerIdx<latterIdx;formerIdx++) {
						Entity* formerSpan = &spanEntities[formerIdx];

						if(formerSpan->labelSpans[0][0] != HB && formerSpan->labelSpans[0][0] != DB)
							continue;

						// find entities that contain former span
						vector< Entity* > entityContainFormerSpan;
						for(int entityIdx=0;entityIdx<entities.size();entityIdx++) {
							Entity* entity = entities[entityIdx];

							for(int spanIdx=0;spanIdx<entity->spans.size();spanIdx++) {
								pair<int, int> & span = entity->spans[spanIdx];

								if(span.first<=formerSpan->spans[0].first && span.second>=formerSpan->spans[0].second) {
									entityContainFormerSpan.push_back(entity);
									break;
								}
							}
						}


						//determine their relation by entities
						Example eg;
						generateOneRelExample(eg, sent, formerSpan, latterSpan, entityContainFormerSpan,
								entityContainLatterSpan,
								false);



						examples.push_back(eg);
						if(eg.m_labels_relation[0]==1) {
							countNegative++;
						} else {
							countPositive++;
						}




					}

				}



			} // sent


		} // doc

		cout<<"positive example: "<< countPositive*1.0/(countPositive+countNegative)<<endl;
		cout<<"negative example: "<< countNegative*1.0/(countPositive+countNegative)<<endl;


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

				int seq_size = labelIdx.size();
				vector<string> outputs;
				outputs.resize(seq_size);
				for (int idx = 0; idx < seq_size; idx++) {
					outputs[idx] = m_driver._modelparams.labelAlpha.from_id(labelIdx[idx], O);
				}

				vector<Entity> spanEntities;
				decode(sent, outputs, spanEntities);

				// use a table to record the relations of all pairs of spans
				vector< vector<int> > relationTable;
				relationTable.resize(spanEntities.size());
				for (int idx = 0; idx < spanEntities.size(); idx++){
					for (int i = 0; i < spanEntities.size(); i++)
						relationTable[idx].push_back(RellabelName2labelID(N));
				}

				// for each pair of spanEntities (HB HI) or (DB DI), determine their relations
				for(int latterIdx=0;latterIdx<spanEntities.size();latterIdx++) {
					Entity* latterSpan = &(spanEntities[latterIdx]);
					vector< Entity* > entityContainLatterSpan;

					if(latterSpan->labelSpans[0][0] != HB && latterSpan->labelSpans[0][0] != DB)
						continue;


					for(int formerIdx=0;formerIdx<latterIdx;formerIdx++) {
						Entity* formerSpan = &(spanEntities[formerIdx]);
						vector< Entity* > entityContainFormerSpan;

						if(formerSpan->labelSpans[0][0] != HB && formerSpan->labelSpans[0][0] != DB)
							continue;

						Example egRel;
						generateOneRelExample(egRel, sent, formerSpan, latterSpan, entityContainFormerSpan,
								entityContainLatterSpan,
								true);

						int result = -1;
						m_driver_relation.predict(egRel, result);

						// the lower triangular will be filled
						relationTable[latterIdx][formerIdx] = result;
					}
				}

				// combine spans to form the final entities according to the table
				// each graph corresponds to an entity
				vector< set<int> > graphs = getCompleteGraph(relationTable);
				vector<Entity> postEntities;

				for(int graphIdx=0;graphIdx<graphs.size();graphIdx++) {
					set<int> & graph = graphs[graphIdx];

					vector<Entity> temp;
					set<int>::iterator iter = graph.begin();
					for(;iter!=graph.end();iter++) {
						int spanIdx = *iter;
						temp.push_back(spanEntities[spanIdx]);
					}

					Entity combined;
					combineAllEntity(temp, combined);
					postEntities.push_back(combined);
				}

				// check each (HB HI), if it occurs in only one entity, it must be a nested entities.
				for(int idx=0;idx<spanEntities.size();idx++) {
					Entity* spanEntity = &(spanEntities[idx]);
					vector< Entity* > entityContainSpan;

					if(spanEntity->labelSpans[0][0] != HB)
						continue;

					for(int entityIdx=0;entityIdx<postEntities.size();entityIdx++) {
						Entity & entity = postEntities[entityIdx];

						for(int spanIdx=0;spanIdx<entity.spans.size();spanIdx++) {
							pair<int, int> & span = entity.spans[spanIdx];

							if(span.first<=spanEntity->spans[0].first && span.second>=spanEntity->spans[0].second) {
								entityContainSpan.push_back(&entity);
								break;
							}
						}
					}

					if(entityContainSpan.size()<=1) {
						postEntities.push_back(*spanEntity);
					}
				}

				// resort by start position and remove the same entity
				vector<Entity> anwserEntities;
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

				// evaluate by ourselves
				ctGoldEntity += entities.size();
				ctPredictEntity += anwserEntities.size();
				for(int i=0;i<anwserEntities.size();i++) {

					int j=0;
					for(;j<entities.size();j++) {
						if(anwserEntities[i].equalsBoundary(*entities[j])) {
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
			vector<Entity> outputEntities;

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

				int seq_size = labelIdx.size();
				vector<string> outputs;
				outputs.resize(seq_size);
				for (int idx = 0; idx < seq_size; idx++) {
					outputs[idx] = m_driver._modelparams.labelAlpha.from_id(labelIdx[idx], O);
				}

				vector<Entity> spanEntities;
				decode(sent, outputs, spanEntities);

				// use a table to record the relations of all pairs of spans
				vector< vector<int> > relationTable;
				relationTable.resize(spanEntities.size());
				for (int idx = 0; idx < spanEntities.size(); idx++){
					for (int i = 0; i < spanEntities.size(); i++)
						relationTable[idx].push_back(RellabelName2labelID(N));
				}
				// for each pair of spanEntities, determine their relations
				for(int latterIdx=0;latterIdx<spanEntities.size();latterIdx++) {
					Entity* latterSpan = &(spanEntities[latterIdx]);
					vector< Entity* > entityContainLatterSpan;

					if(latterSpan->labelSpans[0][0] != HB && latterSpan->labelSpans[0][0] != DB)
						continue;


					for(int formerIdx=0;formerIdx<latterIdx;formerIdx++) {
						Entity* formerSpan = &(spanEntities[formerIdx]);
						vector< Entity* > entityContainFormerSpan;

						if(formerSpan->labelSpans[0][0] != HB && formerSpan->labelSpans[0][0] != DB)
							continue;

						Example egRel;
						generateOneRelExample(egRel, sent, formerSpan, latterSpan, entityContainFormerSpan,
								entityContainLatterSpan,
								true);

						int result = -1;
						m_driver_relation.predict(egRel, result);

						// the lower triangular will be filled
						relationTable[latterIdx][formerIdx] = result;
					}
				}

				// combine spans to form the final entities according to the table
				// each graph corresponds to an entity
				vector< set<int> > graphs = getCompleteGraph(relationTable);
				vector<Entity> postEntities;

				for(int graphIdx=0;graphIdx<graphs.size();graphIdx++) {
					set<int> & graph = graphs[graphIdx];

					vector<Entity> temp;
					set<int>::iterator iter = graph.begin();
					for(;iter!=graph.end();iter++) {
						int spanIdx = *iter;
						temp.push_back(spanEntities[spanIdx]);
					}

					Entity combined;
					combineAllEntity(temp, combined);
					postEntities.push_back(combined);
				}

				// check each (HB HI), if it occurs in only one entity, it must be a nested entities.
				for(int idx=0;idx<spanEntities.size();idx++) {
					Entity* spanEntity = &(spanEntities[idx]);
					vector< Entity* > entityContainSpan;

					if(spanEntity->labelSpans[0][0] != HB)
						continue;

					for(int entityIdx=0;entityIdx<postEntities.size();entityIdx++) {
						Entity & entity = postEntities[entityIdx];

						for(int spanIdx=0;spanIdx<entity.spans.size();spanIdx++) {
							pair<int, int> & span = entity.spans[spanIdx];

							if(span.first<=spanEntity->spans[0].first && span.second>=spanEntity->spans[0].second) {
								entityContainSpan.push_back(&entity);
								break;
							}
						}
					}

					if(entityContainSpan.size()<=1) {
						postEntities.push_back(*spanEntity);
					}
				}

				// resort by start position and remove the same entity
				vector<Entity> anwserEntities;
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


				// evaluate by ourselves
				ctGoldEntity += entities.size();
				ctPredictEntity += anwserEntities.size();
				for(int i=0;i<anwserEntities.size();i++) {

					int j=0;
					for(;j<entities.size();j++) {
						if(anwserEntities[i].equalsBoundary(*entities[j])) {
							ctCorrectEntity ++;
							break;

						}
					}


					outputEntities.push_back(anwserEntities[i]);
				}

			} // sent


			outputResults(doc, outputEntities, m_options.output);

		} // doc

		metric_dev.overall_label_count = ctGoldEntity;
		metric_dev.predicated_label_count = ctPredictEntity;
		metric_dev.correct_label_count = ctCorrectEntity;
		metric_dev.print();

	}


	void decode(fox::Sent& sent, vector<string>& outputs, vector<Entity> & anwserEntities) {

		for(int idx=0;idx<sent.tokens.size();idx++) {
			string& labelName = outputs[idx];
			fox::Token& token = sent.tokens[idx];

			// decode entity label
			if(labelName == B || labelName == HB || labelName == DB) {
				Entity entity;
				newEntity(sent, idx, labelName, entity, 0);
				anwserEntities.push_back(entity);
			} else if(labelName == I || labelName == HI || labelName == DI) {
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
		if(currentLabel!=I && currentLabel!=HI && currentLabel!=DI)
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
			} else {
				if(positionNew==-1 && labelSequence[j]==DB) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=DI) {
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

		eg.isRelation = false;


	}

	void generateOneRelExample(Example& eg, fox::Sent& sent, Entity* formerSpan, Entity* latterSpan,
			vector< Entity* > & entityContainFormerSpan, vector< Entity* > & entityContainLatterSpan,
			bool bPredict) {
		if(!bPredict) {
			string labelName = N;

			bool intersected = false;
			for(int i=0;i<entityContainFormerSpan.size();i++) {
				for(int j=0;j<entityContainLatterSpan.size();j++) {
					if(entityContainFormerSpan[i]->equalsBoundary(*entityContainLatterSpan[j])) {
						intersected = true;
						goto OUT;
					}
				}
			}

OUT:		labelName = intersected ? Y:N;


			int labelID = RellabelName2labelID(labelName);

			for(int i=0;i<MAX_RELATION;i++)
				eg.m_labels_relation.push_back(0.0);
			eg.m_labels_relation[labelID] = 1.0;

		}

		for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
			fox::Token& token = sent.tokens[tokenIdx];

			Feature featureForThisToken;
			featureForThisToken.words.push_back(feature_word(token));

			eg.m_features.push_back(featureForThisToken);

		} // token

#if REL_SEQ

		eg.formerStart = formerSpan->tkSpans[0].first;
		eg.formerEnd = formerSpan->tkSpans[0].second;
		eg.latterStart = latterSpan->tkSpans[0].first;
		eg.latterEnd = latterSpan->tkSpans[0].second;

		for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {

			eg._idxForSeq1.push_back(tokenIdx);

		} // token

#elif REL_SDP
		// use SDP based on the last word of the entity
		vector<int> sdpA;
		vector<int> sdpB;
		int common = fox::Dependency::getCommonAncestor(sent.tokens, formerSpan->tkSpans[0].second, latterSpan->tkSpans[0].second,
				sdpA, sdpB);

		assert(common!=-1); // common ancestor is root

		if(common == -2) {
			eg._idxForSeq1.push_back(formerSpan->tkSpans[0].second);
			eg._idxForSeq1.push_back(latterSpan->tkSpans[0].second);
		} else {

			for(int sdpANodeIdx=0;sdpANodeIdx<sdpA.size();sdpANodeIdx++) { // from a to ancestor(inlcude)
				eg._idxForSeq1.push_back(sdpA[sdpANodeIdx]-1);
			}

			for(int sdpBNodeIdx=sdpB.size()-2;sdpBNodeIdx>=0;sdpBNodeIdx--) { // from ancestor(exclude) to b
				eg._idxForSeq1.push_back(sdpB[sdpBNodeIdx]-1);
			}
		}

#elif REL_BIDT

		// use SDP based on the last word of the entity
		vector<int> sdpA;
		vector<int> sdpB;
		int common = fox::Dependency::getCommonAncestor(sent.tokens, formerSpan->tkSpans[0].second, latterSpan->tkSpans[0].second,
				sdpA, sdpB);

		assert(common!=-1); // common ancestor is root

		if(common == -2) {
			eg._idxForSeq1.push_back(formerSpan->tkSpans[0].second);
			eg._idxForSeq2.push_back(latterSpan->tkSpans[0].second);
		} else {
			for(int sdpANodeIdx=0;sdpANodeIdx<sdpA.size();sdpANodeIdx++) {
				eg._idxForSeq1.push_back(sdpA[sdpANodeIdx]-1);
			}

			for(int sdpBNodeIdx=0;sdpBNodeIdx<sdpB.size();sdpBNodeIdx++) {
				eg._idxForSeq2.push_back(sdpB[sdpBNodeIdx]-1);
			}
		}

#endif



		eg.isRelation = true;

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

		stat2Alphabet(word_stat, m_driver._modelparams.wordAlpha, "ner word", m_options.wordCutOff);

		stat2Alphabet(word_stat, m_driver_relation._modelparams.wordAlpha, "rel word", m_options.wordCutOff);

	}









};



#endif /* NNBB3_H_ */

