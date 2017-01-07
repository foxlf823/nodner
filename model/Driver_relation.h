/*
 * Driver.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_Driver_relation_H_
#define SRC_Driver_relation_H_

#include <iostream>
#include "ComputionGraph_relation.h"


//A native neural network classfier using only word embeddings

class Driver_relation{
public:
	Driver_relation() {
		_pcg = NULL;
	}

	~Driver_relation() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	ComputionGraph_relation *_pcg;  // build neural graphs
	ModelParams_relation _modelparams;  // model parameters
	HyperParams _hyperparams;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;  // model update


public:
	//embeddings are initialized before this separately.
	inline void initial() {
		if (!_hyperparams.bValid()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_modelparams.exportModelParams(_ada);
		_modelparams.exportCheckGradParams(_checkgrad);

		_hyperparams.print();

		_pcg = new ComputionGraph_relation();
		_pcg->createNodes(ComputionGraph_relation::max_sentence_length, _modelparams.types.size());
		_pcg->initial(_modelparams, _hyperparams);

		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
	}


	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		static vector<PMat> tpmats;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example, true);

			//loss function
			int seq_size = example.m_features.size();
			//for (int idx = 0; idx < seq_size; idx++) {
				//cost += _loss.loss(&(_pcg->output[idx]), example.m_labels[idx], _eval, example_num);				
			//}
			cost += _modelparams.loss_relation.loss(&(_pcg->output_relation), example.m_labels_relation, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const Example& eg, int& results) {
		const vector<Feature>& features = eg.m_features;
		_pcg->forward(eg);
		int seq_size = features.size();
		//results.resize(seq_size);
		//for (int idx = 0; idx < seq_size; idx++) {
		//	_loss.predict( &(_pcg->output[idx]), results[idx]);
		//}
		_modelparams.loss_relation.predict(&(_pcg->output_relation), results);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		//loss function
		//for (int idx = 0; idx < seq_size; idx++) {
		//	cost += _loss.cost(&(_pcg->output[idx]), example.m_labels[idx], 1);
		//}
		cost += _modelparams.loss_relation.cost(&(_pcg->output_relation), example.m_labels_relation, 1);

		return cost;
	}


	void updateModel() {
		//_ada.update();
		_ada.update(5.0);
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void writeModel();

	void loadModel();



private:
	inline void resetEval() {
		_eval.reset();
	}


	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_Driver_H_ */
