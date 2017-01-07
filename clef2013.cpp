
#include <vector>
#include "utils.h"
#include <iostream>
#include "Argument_helper.h"
#include "Options.h"
#include "Tool.h"


#include "NNclef.h"
#include "NNclef1.h"
#include "NNclef2.h"
#include "NNclef3.h"
#include "NNclef4.h"



using namespace std;


int main(int argc, char **argv)
{



	string optionFile;
	string trainFile;
	string devFile;
	string testFile;
	string outputFile;
	string trainNlpFile;
	string devNlpFile;
	string testNlpFile;



	dsr::Argument_helper ah;
	ah.new_named_string("train", "", "", "", trainFile);
	ah.new_named_string("dev", "", "", "", devFile);
	ah.new_named_string("test", "", "", "", testFile);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("output", "", "", "", outputFile);
	ah.new_named_string("trainnlp", "", "", "", trainNlpFile);
	ah.new_named_string("devnlp", "", "", "", devNlpFile);
	ah.new_named_string("testnlp", "", "", "", testNlpFile);


	ah.process(argc, argv);
	cout<<"train file: " <<trainFile <<endl;
	cout<<"dev file: "<<devFile<<endl;
	cout<<"test file: "<<testFile<<endl;

	cout<<"trainnlp file: "<<trainNlpFile<<endl;
	cout<<"devnlp file: "<<devNlpFile<<endl;
	cout<<"testnlp file: "<<testNlpFile<<endl;


	Options options;
	options.load(optionFile);

	if(!outputFile.empty())
		options.output = outputFile;

	options.showOptions();

	Tool tool(options);


	//NNclef nn(options);
	//NNclef1 nn(options);
	//NNclef2 nn(options);
	//NNclef3 nn(options);
	NNclef4 nn(options);
	nn.trainAndTest(trainFile, devFile, testFile, tool,
				trainNlpFile, devNlpFile, testNlpFile);



    return 0;

}

