
#ifndef GENIA_H_
#define GENIA_H_

#include <set>
#include <algorithm>
#include "globalsetting.h"

using namespace std;


#define GENIA_PROTEIN 0
#define GENIA_DNA 0
#define GENIA_RNA 0
#define GENIA_CELL_LINE 0
#define GENIA_CELL_TYPE 1
#define GENIA_ALL 0 // only used for debug

// assume that table is lower triangular matrix, and values are 0 or 1
vector< set<int> > getCompleteGraph(vector< vector<int> > & table) {

	map<int, set<int> > relationMap;
	for(int i=0;i<table.size();i++) {
		for(int j=0;j<i;j++) {
			if(table[i][j]==1) {
				// add relations based on i
				map<int, set<int> >::iterator it = relationMap.find(i);
				if (it != relationMap.end()) {
					it->second.insert(j);
				} else {
					set<int> relationNode;
					relationNode.insert(j);
					relationMap.insert(map<int, set<int> >::value_type(i, relationNode));
				}
				// add relations based on j
				it = relationMap.find(j);
				if (it != relationMap.end()) {
					it->second.insert(i);
				} else {
					set<int> relationNode;
					relationNode.insert(i);
					relationMap.insert(map<int, set<int> >::value_type(j, relationNode));
				}
			}
		}
	}

	vector< set<int> > graphs; // each set<int> denotes a complete graph
	// in the initial state, each node is a complete graph
	for(int i=0;i<table.size();i++) {
		set<int> s;
		s.insert(i);
		graphs.push_back(s);
	}

	bool changed = false;
	do {
		changed = false;

		vector< set<int> >::iterator graph = graphs.begin();
		for(; graph!=graphs.end();graph++) {
			// graph e.g. {0,1}
			set<int> intersect; // record the nodes connected with all the nodes in this graph
			bool isolateNode = false;

			set<int>::iterator iterTargetNode = graph->begin();
			for(; iterTargetNode!=graph->end();iterTargetNode++) {
				int node = *iterTargetNode;

				map<int, set<int> >::iterator iterRelationMap = relationMap.find(node);
				if (iterRelationMap == relationMap.end()) { // no nodes connected with this node
					isolateNode = true;
					break;
				}

				set<int> & relationMapOfThisNode = iterRelationMap->second;

				if(iterTargetNode == graph->begin()) { // if first, add it directly
					set<int>::iterator iterTemp=relationMapOfThisNode.begin();
					for(;iterTemp!=relationMapOfThisNode.end();iterTemp++)
						intersect.insert(*iterTemp);
				} else {
					vector<int> temp(table.size(), -1);
					set_intersection(intersect.begin(), intersect.end(), relationMapOfThisNode.begin(), relationMapOfThisNode.end(), temp.begin());
					intersect.clear();
					for(int i=0;i<temp.size();i++) {
						if(temp[i] == -1)
							continue;
						intersect.insert(temp[i]);
					}

					if(intersect.empty()) // we don't need to consider the subsequent nodes
						break;
				}
			}

			if(isolateNode) {
				// do nothing
			} else {
				if(intersect.empty()) {
					// do nothing
				} else {
					// add one node to this graph
					graph->insert(*intersect.begin());
					changed = true;
				}
			}

		} // graph

		if(changed == true) {
			// removed the same graph and get ready for the next iteration
			vector< set<int> > temp;

			for(int i=0;i<graphs.size();i++) {
				set<int> & old = graphs[i];

				bool same = false;
				for(int j=0;j<temp.size();j++) {
					set<int> & new_ = temp[j];

					if(old == new_) {
						same = true;
						break;
					}
				}

				if(same==false) {
					temp.push_back(old);
				}
			}

			graphs.clear();
			for(int i=0;i<temp.size();i++)
				graphs.push_back(temp[i]);

		}


	} while(changed);



	return graphs;
}

// actually, the entities are spans; combine a and b into c
void combineTwoEntity(Entity & a, Entity & b, Entity & c) {

	c.type = a.type; // or b type?

	if(a.spans[0].first<b.spans[0].first) {
		if(a.tkSpans[0].second+1 == b.tkSpans[0].first) { // there are continous spans
			pair<int, int> span(a.spans[0].first, b.spans[0].second);
			c.spans.push_back(span);

			int whitespacetoAdd = b.spans[0].first-a.spans[0].second;
			string textSpan = a.textSpans[0];
			for(int j=0;j<whitespacetoAdd;j++)
				textSpan += " ";
			textSpan += b.textSpans[0];
			c.textSpans.push_back(textSpan);

			pair<int, int> tkSpan(a.tkSpans[0].first, b.tkSpans[0].second);
			c.tkSpans.push_back(tkSpan);
			vector<string> labelSpan;
			for(int i=0;i<a.labelSpans[0].size();i++)
				labelSpan.push_back(a.labelSpans[0][i]);
			for(int i=0;i<b.labelSpans[0].size();i++)
				labelSpan.push_back(b.labelSpans[0][i]);
			c.labelSpans.push_back(labelSpan);

		} else {
			c.spans.push_back(a.spans[0]);
			c.spans.push_back(b.spans[0]);
			c.textSpans.push_back(a.textSpans[0]);
			c.textSpans.push_back(b.textSpans[0]);
			c.tkSpans.push_back(a.tkSpans[0]);
			c.tkSpans.push_back(b.tkSpans[0]);
			c.labelSpans.push_back(a.labelSpans[0]);
			c.labelSpans.push_back(b.labelSpans[0]);
		}

	} else {
		if(b.tkSpans[0].second+1 == a.tkSpans[0].first) { // there are continous spans
			pair<int, int> span(b.spans[0].first, a.spans[0].second);
			c.spans.push_back(span);

			int whitespacetoAdd = a.spans[0].first-b.spans[0].second;
			string textSpan = b.textSpans[0];
			for(int j=0;j<whitespacetoAdd;j++)
				textSpan += " ";
			textSpan += a.textSpans[0];
			c.textSpans.push_back(textSpan);

			pair<int, int> tkSpan(b.tkSpans[0].first, a.tkSpans[0].second);
			c.tkSpans.push_back(tkSpan);
			vector<string> labelSpan;
			for(int i=0;i<b.labelSpans[0].size();i++)
				labelSpan.push_back(b.labelSpans[0][i]);
			for(int i=0;i<a.labelSpans[0].size();i++)
				labelSpan.push_back(a.labelSpans[0][i]);
			c.labelSpans.push_back(labelSpan);
		} else {
			c.spans.push_back(b.spans[0]);
			c.spans.push_back(a.spans[0]);
			c.textSpans.push_back(b.textSpans[0]);
			c.textSpans.push_back(a.textSpans[0]);
			c.tkSpans.push_back(b.tkSpans[0]);
			c.tkSpans.push_back(a.tkSpans[0]);
			c.labelSpans.push_back(b.labelSpans[0]);
			c.labelSpans.push_back(a.labelSpans[0]);
		}
	}

}

// combine all the spans of a into c, assume a is ordered
void combineAllEntity(vector<Entity> & a, Entity & c) {

	Entity lastEntity;
	for(int i=0;i<a.size();i++) {
		Entity temp;
		if(i==0) {
			temp.spans.push_back(a[i].spans[0]);
			temp.textSpans.push_back(a[i].textSpans[0]);
			temp.tkSpans.push_back(a[i].tkSpans[0]);
			temp.labelSpans.push_back(a[i].labelSpans[0]);
		} else {
			combineTwoEntity(lastEntity, a[i], temp);
		}
		lastEntity = temp;

	}

	c = lastEntity;
	c.type = a[0].type;
}

// can only be used for continuous ones
bool boolTokenInSpan(const fox::Token& tok, pair<int, int> & span) {

	if(tok.begin>=span.first && tok.end<=span.second)
		return true;
	else
		return false;
}

void findEntityInSent(Sentence & sent, vector<Entity*> & results) {

	for(int i=0;i<sent.entities.size();i++) {
		results.push_back(&(sent.entities[i]));
	}

	return;
}

void fillTokenIdx(vector<Document> & documents) {

	for(int docIdx=0;docIdx<documents.size();docIdx++) {
		Document& doc = documents[docIdx];

		for(int sentIdx=0;sentIdx<doc.sentences.size();sentIdx++) {
			Sentence & sent = doc.sentences[sentIdx];

			/*vector<Entity*> entities;
			findEntityInSent(sent, entities);*/
			vector<Entity>::iterator iter = sent.entities.begin();

			for(;iter!=sent.entities.end();) {
				bool deleteThis = false;

				for(int spanIdx=0;spanIdx<iter->spans.size();spanIdx++) {
					pair<int, int> & span = iter->spans[spanIdx];
					pair<int, int> tkSpan;
					tkSpan.first = -1;
					tkSpan.second = -1;

					for(int tokenIdx=0;tokenIdx<sent.info.tokens.size();tokenIdx++) {
						const fox::Token& token = sent.info.tokens[tokenIdx];

						if(boolTokenInSpan(token, span)) {
							tkSpan.first = tokenIdx;
							break;
						}
					}

					for(int tokenIdx=sent.info.tokens.size()-1;tokenIdx>=0;tokenIdx--) {
						const fox::Token& token = sent.info.tokens[tokenIdx];

						if(boolTokenInSpan(token, span)) {
							tkSpan.second = tokenIdx;
							break;
						}
					}

					if(tkSpan.first==-1 || tkSpan.second==-1) {
						deleteThis = true;
						break;
					}
					//assert(tkSpan.first!=-1);
					//assert(tkSpan.second!=-1);
					else
						iter->tkSpans.push_back(tkSpan);



				} // span

				if(deleteThis)
					iter = sent.entities.erase(iter);
				else
					iter++;

			} // entity


		} // sent


	} // doc

	return;
}

void loadNlpFile(const string& foldPath, vector< vector<Document> >& groups) {

	for(int ctFold=0;ctFold<groups.size();ctFold++) {
		stringstream ss;
		ss<<foldPath<<"/"<<ctFold;
		string dirPath = ss.str();
		vector<Document>& group = groups[ctFold];

		struct dirent** namelist = NULL;
		int total = scandir(dirPath.c_str(), &namelist, 0, alphasort);
		int count = 0;

		for(int i=0;i<total;i++) {

			if (namelist[i]->d_type == 8) {
				//file
				if(namelist[i]->d_name[0]=='.')
					continue;

				string filePath = dirPath;
				filePath += "/";
				filePath += namelist[i]->d_name;
				string fileName = namelist[i]->d_name;

				ifstream ifs;
				ifs.open(filePath.c_str());

				Document& doc = group[count];

				string line;
				int sentenceCount = 0;


				while(getline(ifs, line)) {
					if(line.empty()){
						fox::Sent& sent = doc.sentences[sentenceCount].info;
						sent.begin = sent.tokens[0].begin;
						sent.end = sent.tokens[sent.tokens.size()-1].end;

						sentenceCount++;
					} else {
						fox::Sent& sent = doc.sentences[sentenceCount].info;
						vector<string> splitted;
						fox::split_bychar(line, splitted, '\t');
						fox::Token token;
						token.word = splitted[0];
						token.begin = atoi(splitted[1].c_str());
						token.end = atoi(splitted[2].c_str());
						token.pos = splitted[3];
						token.depGov = atoi(splitted[4].c_str());
						token.depType = splitted[5];
						sent.tokens.push_back(token);
					}



				}

				ifs.close();
				count++;
			}
		}
	}


	return;

}

int loadAnnotatedFile(const string& foldPath, vector< vector<Document> >& groups)
{

	for(int ctFold=0;ctFold<groups.size();ctFold++) {
		stringstream ss;
		ss<<foldPath<<"/"<<ctFold;
		string dirPath = ss.str();
		vector<Document>& group = groups[ctFold];

		struct dirent** namelist = NULL;
		int total = scandir(dirPath.c_str(), &namelist, 0, alphasort);

		for(int i=0;i<total;i++) {

			if (namelist[i]->d_type == 8) {
				//file
				if(namelist[i]->d_name[0]=='.')
					continue;

				string filePath = dirPath;
				filePath += "/";
				filePath += namelist[i]->d_name;
				string fileName = namelist[i]->d_name;

				Document doc;
				int fileSuffixPosition = fileName.find_first_of(".txt");
				doc.id += fileName.substr(0, fileSuffixPosition);

				ifstream ifs;
				ifs.open(filePath.c_str());
				string line;
				while(getline(ifs, line)) {
					if(string::npos != line.find("#S#")) { // a sentence begin
						Sentence sentence;

						doc.sentences.push_back(sentence);
					} else if(!line.empty()) { // mention annotation
						Sentence& sentence = doc.sentences[doc.sentences.size()-1];

						vector<string> splitted;
						fox::split_bystring(line, splitted, "\t");
#if GENIA_PROTEIN
						if(splitted[0] != "protein")
							continue;
#elif GENIA_DNA
						if(splitted[0] != "DNA")
							continue;
#elif GENIA_RNA
						if(splitted[0] != "RNA")
							continue;
#elif GENIA_CELL_LINE
						if(splitted[0] != "cell_line")
							continue;
#elif GENIA_CELL_TYPE
						if(splitted[0] != "cell_type")
							continue;
#elif GENIA_ALL

#else
						assert(0);
#endif

						Entity entity;
						entity.type = splitted[0];
						pair<int, int> span(atoi(splitted[1].c_str()), atoi(splitted[2].c_str()));
						entity.spans.push_back(span);

						sentence.entities.push_back(entity);

					}


				}

				ifs.close();

				group.push_back(doc);


			}
		}

	}






    return 0;

}


#endif
