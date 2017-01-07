#ifndef SRC_ComputionGraph_joint_H_
#define SRC_ComputionGraph_joint_H_

#include "ModelParams_joint.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph_joint : Graph{
public:
	const static int max_sentence_length = MAX_SENTENCE_LENGTH;

public:
	// node instances
	vector<vector<LookupNode> > word_inputs;


	vector<ConcatNode> token_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;

	vector<BiNode> word_hidden2;
	vector<LinearNode> output;

	int type_num;

	//dropout nodes
	vector<vector<DropNode> > word_inputs_drop;
	vector<DropNode> word_hidden1_drop;
	vector<DropNode> word_hidden2_drop;

#if REL_SEQ || REL_SDP
	vector<ConcatNode> lstm_relation_inputs;
	LSTM1Builder left_lstm_relation;
	LSTM1Builder right_lstm_relation;


	ConcatNode former_entity_repsent;
	ConcatNode latter_entity_repsent;
#elif REL_BIDT
	vector<ConcatNode> lstm_relation_inputs_seq1;
	vector<ConcatNode> lstm_relation_inputs_seq2;
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
	ComputionGraph_joint() : Graph(){
	}

	~ComputionGraph_joint(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int typeNum){
		type_num = typeNum;
		resizeVec(word_inputs, sent_length, type_num + 1);


		token_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);
		word_hidden2.resize(sent_length);
		output.resize(sent_length);

		resizeVec(word_inputs_drop, sent_length, type_num + 1);
		word_hidden1_drop.resize(sent_length);
		word_hidden2_drop.resize(sent_length);

#if REL_SEQ || REL_SDP
		lstm_relation_inputs.resize(sent_length);
		left_lstm_relation.resize(sent_length);
		right_lstm_relation.resize(sent_length);


#elif REL_BIDT
		lstm_relation_inputs_seq1.resize(sent_length);
		lstm_relation_inputs_seq2.resize(sent_length);
		lstm_relation_seq1_bottomup.resize(sent_length);
		lstm_relation_seq2_bottomup.resize(sent_length);
		lstm_relation_seq1_topdown.resize(sent_length);
		lstm_relation_seq2_topdown.resize(sent_length);
#endif
	}

	inline void clear(){
		Graph::clear();
		clearVec(word_inputs);


		word_hidden1.clear();
		left_lstm.clear();
		right_lstm.clear();
		word_hidden2.clear();
		output.clear();

		clearVec(word_inputs_drop);
		word_hidden1_drop.clear();
		word_hidden2_drop.clear();

#if REL_SEQ || REL_SDP
		lstm_relation_inputs.clear();
		left_lstm_relation.clear();
		right_lstm_relation.clear();


#elif REL_BIDT
		lstm_relation_inputs_seq1.clear();
		lstm_relation_inputs_seq2.clear();
		lstm_relation_seq1_bottomup.clear();
		lstm_relation_seq2_bottomup.clear();
		lstm_relation_seq1_topdown.clear();
		lstm_relation_seq2_topdown.clear();
#endif
	}

public:
	inline void initial(ModelParams_joint& model, HyperParams& opts){
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&model.words);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&model.types[idy - 1]);
			}

			for (int idy = 0; idy < word_inputs[idx].size(); idy++){
				word_inputs_drop[idx][idy].setDropValue(opts.dropOut);
			}


			word_hidden1[idx].setParam(&model.tanh1_project);
			word_hidden1[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden1_drop[idx].setDropValue(opts.dropOut);

			word_hidden2[idx].setParam(&model.tanh2_project);
			word_hidden2[idx].setFunctions(&tanh, &tanh_deri);
			word_hidden2_drop[idx].setDropValue(opts.dropOut);
		}
		word_window.setContext(opts.wordcontext);
		left_lstm.setParam(&model.left_lstm_project, opts.dropOut, true);
		right_lstm.setParam(&model.right_lstm_project, opts.dropOut, false);

		for (int idx = 0; idx < output.size(); idx++){
			output[idx].setParam(&model.olayer_linear);
		}

#if REL_SEQ || REL_SDP
		left_lstm_relation.setParam(&model.left_lstm_project_relation, opts.dropOut, true);
		right_lstm_relation.setParam(&model.right_lstm_project_relation, opts.dropOut, false);

#elif REL_BIDT
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
	inline void forward_entity(const vector<Feature>& features, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		// second step: build graph
		int seq_size = features.size();
		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx][0].forward(this, feature.words[0]);

			//drop out
			word_inputs_drop[idx][0].forward(this, &word_inputs[idx][0]);

			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].forward(this, feature.types[idy - 1]);
				//drop out
				word_inputs_drop[idx][idy].forward(this, &word_inputs[idx][idy]);
			}


			token_repsents[idx].forward(this, getPNodes(word_inputs_drop[idx], word_inputs_drop[idx].size()));

		}

		//windowlized
		word_window.forward(this, getPNodes(token_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(this, &(word_window._outputs[idx]));

			word_hidden1_drop[idx].forward(this, &word_hidden1[idx]);
		}

		left_lstm.forward(this, getPNodes(word_hidden1_drop, seq_size));
		right_lstm.forward(this, getPNodes(word_hidden1_drop, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden2[idx].forward(this, &(left_lstm._hiddens_drop[idx]), &(right_lstm._hiddens_drop[idx]));

			word_hidden2_drop[idx].forward(this, &word_hidden2[idx]);

			output[idx].forward(this, &(word_hidden2_drop[idx]));
		}
	}

	inline void forward_relation(const Example & eg, bool bTrain = false){
		const vector<Feature>& features = eg.m_features;

		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation


		// second step: build graph
		int seq_size = features.size();
		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx][0].forward(this, feature.words[0]);

			//drop out
			word_inputs_drop[idx][0].forward(this, &word_inputs[idx][0]);

			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].forward(this, feature.types[idy - 1]);
				//drop out
				word_inputs_drop[idx][idy].forward(this, &word_inputs[idx][idy]);
			}


			token_repsents[idx].forward(this, getPNodes(word_inputs_drop[idx], word_inputs_drop[idx].size()));

		}

		//windowlized
		word_window.forward(this, getPNodes(token_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(this, &(word_window._outputs[idx]));

			word_hidden1_drop[idx].forward(this, &word_hidden1[idx]);
		}

		left_lstm.forward(this, getPNodes(word_hidden1_drop, seq_size));
		right_lstm.forward(this, getPNodes(word_hidden1_drop, seq_size));

#if REL_SEQ
		for (int idx = 0; idx < seq_size; idx++) {
			lstm_relation_inputs[idx].forward(this, &(left_lstm._hiddens_drop[idx]), &(right_lstm._hiddens_drop[idx]));
		}

		left_lstm_relation.forward(this, getPNodes(lstm_relation_inputs, seq_size));
		right_lstm_relation.forward(this, getPNodes(lstm_relation_inputs, seq_size));

		former_entity_repsent.forward(this, &(left_lstm_relation._hiddens_drop[eg.formerEnd]), &(right_lstm_relation._hiddens_drop[eg.formerStart]));
		latter_entity_repsent.forward(this, &(left_lstm_relation._hiddens_drop[eg.latterEnd]), &(right_lstm_relation._hiddens_drop[eg.latterStart]));


		word_hidden2_relation.forward(this, &(former_entity_repsent), &(latter_entity_repsent));
#elif REL_SDP

		int seq_size_relation = eg._idxForSeq1.size();
		for(int i=0; i<seq_size_relation; i++) {
			int idx = eg._idxForSeq1[i];

			lstm_relation_inputs[i].forward(this, &(left_lstm._hiddens_drop[idx]), &(right_lstm._hiddens_drop[idx]));

		}

		left_lstm_relation.forward(this, getPNodes(lstm_relation_inputs, seq_size_relation));
		right_lstm_relation.forward(this, getPNodes(lstm_relation_inputs, seq_size_relation));


		word_hidden2_relation.forward(this, &(left_lstm_relation._hiddens_drop[seq_size_relation-1]), &(right_lstm_relation._hiddens_drop[0]));


#elif REL_BIDT
		int seq_size1 = eg._idxForSeq1.size();
		int seq_size2 = eg._idxForSeq2.size();

		for(int i=0; i<seq_size1; i++) {
			int idx = eg._idxForSeq1[i];
			lstm_relation_inputs_seq1[i].forward(this, &(left_lstm._hiddens_drop[idx]), &(right_lstm._hiddens_drop[idx]));
		}

		for(int i=0; i<seq_size2; i++) {
			int idx = eg._idxForSeq2[i];
			lstm_relation_inputs_seq2[i].forward(this, &(left_lstm._hiddens_drop[idx]), &(right_lstm._hiddens_drop[idx]));
		}

		lstm_relation_seq1_bottomup.forward(this, getPNodes(lstm_relation_inputs_seq1, seq_size1));
		lstm_relation_seq2_bottomup.forward(this, getPNodes(lstm_relation_inputs_seq2, seq_size2));
		lstm_relation_seq1_topdown.forward(this, getPNodes(lstm_relation_inputs_seq1, seq_size1));
		lstm_relation_seq2_topdown.forward(this, getPNodes(lstm_relation_inputs_seq2, seq_size2));

		seq1_repsent.forward(this, &(lstm_relation_seq1_bottomup._hiddens_drop[seq_size1-1]), &(lstm_relation_seq1_topdown._hiddens_drop[0]));
		seq2_repsent.forward(this, &(lstm_relation_seq2_bottomup._hiddens_drop[seq_size2-1]), &(lstm_relation_seq2_topdown._hiddens_drop[0]));
		word_hidden2_relation.forward(this, &(seq1_repsent), &(seq2_repsent));
#endif

		word_hidden2_drop_relation.forward(this, &word_hidden2_relation);

		output_relation.forward(this, &(word_hidden2_drop_relation));
	}

};

#endif /* SRC_ComputionGraph_H_ */
