

#ifndef DOCUMENT_H_
#define DOCUMENT_H_

#include <string>
#include "Entity.h"
#include "Sent.h"
#include "Sentence.h"

using namespace std;


class Document {
public:
	Document() {

	}
/*	virtual ~BiocDocument() {

	}*/

	string id;
	vector<Entity> entities;
	vector<fox::Sent> sents;
	int maxParagraphId;

	// another usage
	vector<Sentence> sentences;
};

#endif /* DOCUMENT_H_ */
