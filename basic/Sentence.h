/*
 * Sentence.h
 *
 *  Created on: Dec 1, 2016
 *      Author: fox
 */

#ifndef SENTENCE_H_
#define SENTENCE_H_

#include <vector>
#include "Sent.h"

using namespace std;

class Sentence {
public:
	Sentence() {

	}
	vector<Entity> entities;
	fox::Sent info;
};

#endif /* SENTENCE_H_ */
