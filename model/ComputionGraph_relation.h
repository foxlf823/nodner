#ifndef SRC_ComputionGraph_relation_H_
#define SRC_ComputionGraph_relation_H_

#include "ModelParams_relation.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph_relation : Graph{
public:
	const static int max_sentence_length = MAX_SENTENCE_LENGTH;

public:

#if REL_SEQ || REL_SDP
	// node instances
	vector<vector<LookupNode> > word_inputs;
	vector<vector<DropNode> > word_inputs_drop;

	vector<ConcatNode> token_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;
	vector<DropNode> word_hidden1_drop;

	ConcatNode former_entity_repsent;
	ConcatNode latter_entity_repsent;

	LSTM1Builder left_lstm_relation;
	LSTM1Builder right_lstm_relation;
#elif REL_BIDT
	vector<vector<LookupNode> > word_inputs_seq1;
	vector<vector<DropNode> > word_inputs_drop_seq1;
	vector<ConcatNode> token_repsents_seq1;
	WindowBuilder word_window_seq1;
	vector<UniNode> word_hidden1_seq1;
	vector<DropNode> word_hidden1_drop_seq1;

	vector<vector<LookupNode> > word_inputs_seq2;
	vector<vector<DropNode> > word_inputs_drop_seq2;
	vector<ConcatNode> token_repsents_seq2;
	WindowBuilder word_window_seq2;
	vector<UniNode> word_hidden1_seq2;
	vector<DropNode> word_hidden1_drop_seq2;

	LSTM1Builder lstm_relation_seq1_bottomup;
	LSTM1Builder lstm_relation_seq2_bottomup;
	LSTM1Builder lstm_relation_seq1_topdown;
	LSTM1Builder lstm_relation_seq2_topdown;
	ConcatNode seq1_repsent;
	ConcatNode seq2_repsent;

#endif



	BiNode word_hidden2_relation;
	DropNode word_hidden2_drop_relation;

	LinearNode output_relation;



	// node pointers
public:
	ComputionGraph_relation() : Graph(){
	}

	~ComputionGraph_relation(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int typeNum){

#if REL_SEQ || REL_SDP


		resizeVec(word_inputs, sent_length, 1);
		resizeVec(word_inputs_drop, sent_length, 1);

		token_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		word_hidden1_drop.resize(sent_length);

		left_lstm_relation.resize(sent_length);
		right_lstm_relation.resize(sent_length);
#elif REL_BIDT


		resizeVec(word_inputs_seq1, sent_length, 1);
		resizeVec(word_inputs_drop_seq1, sent_length, 1);

		token_repsents_seq1.resize(sent_length);
		word_window_seq1.resize(sent_length);
		word_hidden1_seq1.resize(sent_length);
		word_hidden1_drop_seq1.resize(sent_length);


		resizeVec(word_inputs_seq2, sent_length, 1);
		resizeVec(word_inputs_drop_seq2, sent_length, 1);

		token_repsents_seq2.resize(sent_length);
		word_window_seq2.resize(sent_length);
		word_hidden1_seq2.resize(sent_length);
		word_hidden1_drop_seq2.resize(sent_length);

		lstm_relation_seq1_bottomup.resize(sent_length);
		lstm_relation_seq2_bottomup.resize(sent_length);
		lstm_relation_seq1_topdown.resize(sent_length);
		lstm_relation_seq2_topdown.resize(sent_length);

#endif


	}

	inline void clear(){
		Graph::clear();
#if REL_SEQ || REL_SDP
		clearVec(word_inputs);
		clearVec(word_inputs_drop);
		token_repsents.clear();
		word_window.clear();
		word_hidden1.clear();
		word_hidden1_drop.clear();

		left_lstm_relation.clear();
		right_lstm_relation.clear();

#elif REL_BIDT
		clearVec(word_inputs_seq1);
		clearVec(word_inputs_drop_seq1);
		token_repsents_seq1.clear();
		word_window_seq1.clear();
		word_hidden1_seq1.clear();
		word_hidden1_drop_seq1.clear();


		clearVec(word_inputs_seq2);
		clearVec(word_inputs_drop_seq2);
		token_repsents_seq2.clear();
		word_window_seq2.clear();
		word_hidden1_seq2.clear();
		word_hidden1_drop_seq2.clear();


		lstm_relation_seq1_bottomup.clear();
		lstm_relation_seq2_bottomup.clear();
		lstm_relation_seq1_topdown.clear();
		lstm_relation_seq2_topdown.clear();

#endif



	}

public:
	inline void initial(ModelParams_relation& model, HyperParams& opts){
#if REL_SEQ || REL_SDP
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&model.words);
			word_inputs_drop[idx][0].setDropValue(opts.dropOut);


			word_hidden1[idx].setParam(&model.tanh1_project);
			word_hidden1[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden1_drop[idx].setDropValue(opts.dropOut);

		}
		word_window.setContext(opts.wordcontext);

		left_lstm_relation.setParam(&model.left_lstm_project_relation, opts.dropOut, true);
		right_lstm_relation.setParam(&model.right_lstm_project_relation, opts.dropOut, false);
#elif REL_BIDT
		for (int idx = 0; idx < word_inputs_seq1.size(); idx++) {
			word_inputs_seq1[idx][0].setParam(&model.words);
			word_inputs_drop_seq1[idx][0].setDropValue(opts.dropOut);


			word_hidden1_seq1[idx].setParam(&model.tanh1_project);
			word_hidden1_seq1[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden1_drop_seq1[idx].setDropValue(opts.dropOut);

		}
		word_window_seq1.setContext(opts.wordcontext);

		for (int idx = 0; idx < word_inputs_seq2.size(); idx++) {

			word_inputs_seq2[idx][0].setParam(&model.words);
			word_inputs_drop_seq2[idx][0].setDropValue(opts.dropOut);

			word_hidden1_seq2[idx].setParam(&model.tanh1_project);
			word_hidden1_seq2[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden1_drop_seq2[idx].setDropValue(opts.dropOut);

		}
		word_window_seq2.setContext(opts.wordcontext);

		lstm_relation_seq1_bottomup.setParam(&model.left_lstm_project_relation, opts.dropOut, true);
		lstm_relation_seq2_bottomup.setParam(&model.left_lstm_project_relation, opts.dropOut, true);
		lstm_relation_seq1_topdown.setParam(&model.right_lstm_project_relation, opts.dropOut, false);
		lstm_relation_seq2_topdown.setParam(&model.right_lstm_project_relation, opts.dropOut, false);

#endif



		word_hidden2_relation.setParam(&model.tanh2_project_relation);
		word_hidden2_relation.setFunctions(&tanh, &tanh_deri);
		word_hidden2_drop_relation.setDropValue(opts.dropOut);
		output_relation.setParam(&model.olayer_linear_relation);

	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Example & eg, bool bTrain = false){
		const vector<Feature>& features = eg.m_features;

		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

#if REL_SEQ || REL_SDP
		int seq_size = eg._idxForSeq1.size();

		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[eg._idxForSeq1[idx]];

			//input
			word_inputs[idx][0].forward(this, feature.words[0]);

			//drop out
			word_inputs_drop[idx][0].forward(this, &word_inputs[idx][0]);

			token_repsents[idx].forward(this, getPNodes(word_inputs_drop[idx], word_inputs_drop[idx].size()));
		}

		//windowlized
		word_window.forward(this, getPNodes(token_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(this, &(word_window._outputs[idx]));

			word_hidden1_drop[idx].forward(this, &word_hidden1[idx]);
		}

		left_lstm_relation.forward(this, getPNodes(word_hidden1_drop, seq_size));
		right_lstm_relation.forward(this, getPNodes(word_hidden1_drop, seq_size));

#if REL_SEQ
		former_entity_repsent.forward(this, &(left_lstm_relation._hiddens_drop[eg.formerEnd]), &(right_lstm_relation._hiddens_drop[eg.formerStart]));
		latter_entity_repsent.forward(this, &(left_lstm_relation._hiddens_drop[eg.latterEnd]), &(right_lstm_relation._hiddens_drop[eg.latterStart]));
		word_hidden2_relation.forward(this, &(former_entity_repsent), &(latter_entity_repsent));
#elif REL_SDP
		word_hidden2_relation.forward(this, &(left_lstm_relation._hiddens_drop[seq_size-1]), &(right_lstm_relation._hiddens_drop[0]));
#endif

#elif REL_BIDT
		int seq_size1 = eg._idxForSeq1.size();
		int seq_size2 = eg._idxForSeq2.size();

		for (int idx = 0; idx < seq_size1; idx++) {
			const Feature& feature = features[eg._idxForSeq1[idx]];

			//input
			word_inputs_seq1[idx][0].forward(this, feature.words[0]);

			//drop out
			word_inputs_drop_seq1[idx][0].forward(this, &word_inputs_seq1[idx][0]);


			token_repsents_seq1[idx].forward(this, getPNodes(word_inputs_drop_seq1[idx], word_inputs_drop_seq1[idx].size()));
		}

		//windowlized
		word_window_seq1.forward(this, getPNodes(token_repsents_seq1, seq_size1));

		for (int idx = 0; idx < seq_size1; idx++) {
			//feed-forward
			word_hidden1_seq1[idx].forward(this, &(word_window_seq1._outputs[idx]));

			word_hidden1_drop_seq1[idx].forward(this, &word_hidden1_seq1[idx]);
		}

		for (int idx = 0; idx < seq_size2; idx++) {
			const Feature& feature = features[eg._idxForSeq2[idx]];

			//input
			word_inputs_seq2[idx][0].forward(this, feature.words[0]);

			//drop out
			word_inputs_drop_seq2[idx][0].forward(this, &word_inputs_seq2[idx][0]);


			token_repsents_seq2[idx].forward(this, getPNodes(word_inputs_drop_seq2[idx], word_inputs_drop_seq2[idx].size()));
		}

		//windowlized
		word_window_seq2.forward(this, getPNodes(token_repsents_seq2, seq_size2));

		for (int idx = 0; idx < seq_size2; idx++) {
			//feed-forward
			word_hidden1_seq2[idx].forward(this, &(word_window_seq2._outputs[idx]));

			word_hidden1_drop_seq2[idx].forward(this, &word_hidden1_seq2[idx]);
		}


		lstm_relation_seq1_bottomup.forward(this, getPNodes(word_hidden1_drop_seq1, seq_size1));
		lstm_relation_seq2_bottomup.forward(this, getPNodes(word_hidden1_drop_seq2, seq_size2));
		lstm_relation_seq1_topdown.forward(this, getPNodes(word_hidden1_drop_seq1, seq_size1));
		lstm_relation_seq2_topdown.forward(this, getPNodes(word_hidden1_drop_seq2, seq_size2));

		seq1_repsent.forward(this, &(lstm_relation_seq1_bottomup._hiddens_drop[seq_size1-1]), &(lstm_relation_seq1_topdown._hiddens_drop[0]));
		seq2_repsent.forward(this, &(lstm_relation_seq2_bottomup._hiddens_drop[seq_size2-1]), &(lstm_relation_seq2_topdown._hiddens_drop[0]));
		word_hidden2_relation.forward(this, &(seq1_repsent), &(seq2_repsent));


#endif


		word_hidden2_drop_relation.forward(this, &word_hidden2_relation);

		output_relation.forward(this, &(word_hidden2_drop_relation));
	}

};

#endif /* SRC_ComputionGraph_H_ */
