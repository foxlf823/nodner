
#ifndef ENTITY_H_
#define ENTITY_H_

#include <string>
#include <sstream>
#include <utility>

using namespace std;

class Entity {
public:
	  string id; // id is unique in a document
	  string type;

	  // begin, end
	  vector< pair<int, int> > spans;
	  vector< string > textSpans;
	  // tkStart, tkEnd
	  vector< pair<int, int> > tkSpans;
	  vector< vector<string> > labelSpans;

	  int sentIdx; // the sentence index which this entity belongs to

	Entity() {
		id = "-1";
		type = "";


		sentIdx = -1;

	}

	bool equals(const Entity& another) const {
		if(type == another.type && spans.size()==another.spans.size()) {

			for(int i=0;i<spans.size();i++) {
				const pair<int, int> & span = spans[i];
				const pair<int, int> & anotherSpan = another.spans[i];

				if(span.first != anotherSpan.first || span.second != anotherSpan.second)
					return false;
			}

			return true;
		}
		else
			return false;
	}


		bool equalsBoundary(const Entity& another) const {
			if(spans.size()==another.spans.size()) {

				for(int i=0;i<spans.size();i++) {
					const pair<int, int> & span = spans[i];
					const pair<int, int> & anotherSpan = another.spans[i];

					if(span.first != anotherSpan.first || span.second != anotherSpan.second)
						return false;
				}

				return true;
			}
			else
				return false;
		}

		bool equalsType(const Entity& another) const {
			if(type == another.type)
				return true;
			else
				return false;
		}
};



#endif /* ENTITY_H_ */
