/*
 * clef2013.h
 *
 *  Created on: Nov 17, 2016
 *      Author: fox
 */

#ifndef CLEF2013_H_
#define CLEF2013_H_

#include <set>
#include <algorithm>
#include "globalsetting.h"

using namespace std;



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


void findEntityInSent(int begin, int end, Document& doc, vector<Entity*> & results) {

	for(int i=0;i<doc.entities.size();i++) {
		Entity & entity = doc.entities[i];
		bool isIn = true;
		for(int spanIdx=0;spanIdx<entity.spans.size();spanIdx++) {
			pair<int, int> & span = entity.spans[spanIdx];
			if(span.first<begin || span.second>end) {
				isIn = false;
				break;
			}
		}

		if(isIn)
			results.push_back(&(entity));
	}

	return;
}


void fillTokenIdx(vector<Document> & documents) {

	for(int docIdx=0;docIdx<documents.size();docIdx++) {
		Document& doc = documents[docIdx];

		for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
			fox::Sent & sent = doc.sents[sentIdx];

			vector<Entity*> entities;
			findEntityInSent(sent.begin, sent.end, doc, entities);

			for(int entityIdx=0;entityIdx<entities.size();entityIdx++) {
				Entity* entity = entities[entityIdx];

				for(int spanIdx=0;spanIdx<entity->spans.size();spanIdx++) {
					pair<int, int> & span = entity->spans[spanIdx];
					pair<int, int> tkSpan;
					tkSpan.first = -1;
					tkSpan.second = -1;

					for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
						const fox::Token& token = sent.tokens[tokenIdx];

						if(boolTokenInSpan(token, span)) {
							tkSpan.first = tokenIdx;
							break;
						}
					}

					for(int tokenIdx=sent.tokens.size()-1;tokenIdx>=0;tokenIdx--) {
						const fox::Token& token = sent.tokens[tokenIdx];

						if(boolTokenInSpan(token, span)) {
							tkSpan.second = tokenIdx;
							break;
						}
					}

					if(tkSpan.first==-1)
						cout<<endl;
					assert(tkSpan.first!=-1);
					assert(tkSpan.second!=-1);

					entity->tkSpans.push_back(tkSpan);
				} // span



			} // entity


		} // sent


	} // doc

	return;
}


void outputResults(const Document& doc, vector<Entity>& entities, const string& dir) {
	ofstream m_outf;
	string path = dir+"/"+doc.id+".pipe.txt";
	m_outf.open(path.c_str());

	for(int i=0;i<entities.size();i++) {
		Entity & entity = entities[i];

		// 00098-016139-DISCHARGE_SUMMARY.txt||Disease_Disorder||C0221755||1141||1148||1192||1198

		m_outf << doc.id << ".txt"<< "||"<< "Disease_Disorder"<< "||"<<"C0221755";

		for(int spanIdx=0;spanIdx<entity.spans.size();spanIdx++) {
			pair<int, int> & span = entity.spans[spanIdx];
			m_outf<<"||"<< span.first << "||" <<span.second;
		}

		m_outf << endl;

	}

	m_outf.close();
}


void loadNlpFile(const string& dirPath, vector<Document>& docs) {

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
			fox::Sent sent;
			string line;


			while(getline(ifs, line)) {
				if(line.empty()){
					// new line
					if(sent.tokens.size()!=0) {
						docs[count].sents.push_back(sent);
						docs[count].sents[docs[count].sents.size()-1].begin = sent.tokens[0].begin;
						docs[count].sents[docs[count].sents.size()-1].end = sent.tokens[sent.tokens.size()-1].end;
						sent.tokens.clear();
					}
				} else {
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

	return;

}

int loadAnnotatedFile(const string& dirPath, vector<Document>& documents)
{
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

			if(string::npos != filePath.find(".pipe.txt")) {
				Document doc;
				int fileSuffixPosition = fileName.find_first_of(".pipe.txt");
				doc.id += fileName.substr(0, fileSuffixPosition);

				ifstream ifs;
				ifs.open(filePath.c_str());
				string line;
				while(getline(ifs, line)) {
					if(!line.empty()) {

						vector<string> splitted;
						fox::split_bystring(line, splitted, "||");
						Entity entity;
						//splitted[0]
						entity.type = splitted[1];
						//splitted[2]

						int spanNumber = (splitted.size()-3);
						assert(spanNumber%2==0);

						for(int spanIdx=0;spanIdx<spanNumber/2;spanIdx++) {
							pair<int, int> span(atoi(splitted[spanIdx*2+3].c_str()), atoi(splitted[spanIdx*2+1+3].c_str()));
							entity.spans.push_back(span);

						}

						doc.entities.push_back(entity);

					}


				}

				ifs.close();

				documents.push_back(doc);
			}

		}
	}


    return 0;

}


#endif /* CLEF2013_H_ */
