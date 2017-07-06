This is a C++ project for recognizing irreguar entities from biomedical text.
For details, please refer to:
Li F, Zhang M, Tian B, Chen B, Fu G, Ji D. Recognizing irregular entities in biomedical text via deep neural networks [J]. Pattern Recognition Letters, 2017. DOI: https://doi.org/10.1016/j.patrec.2017.06.009

To run the program with sample data of CLEF, please use the following commond line:
clef -option clef2013.option -train yourpath/sample_set -dev yourpath/sample_set -test yourpath/sample_set -trainnlp yourpath/sample_nlp -devnlp yourpath/sample_nlp -testnlp yourpath/sample_nlp -output yourpath/output

This project depends on the external libraries as below:
FoxUtil (https://github.com/foxlf823/FoxUtil)
eigen
LibN3L