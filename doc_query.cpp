#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/SVD>

#define MAX_STRING 50
#define FOUND -1
#define QUERY_PATH "Documents/Machine Learning/MachineLearning/MachineLearning/queries"
#define DOCS_PATH "Documents/Machine Learning/MachineLearning/MachineLearning/docs"

using namespace std;
using namespace Eigen;

bool my_predicate(char c){
    if (c == '\n' || c == '\r' || c == '\0')
        return true;
    else
        return false;
}
std::vector<std::string> getfiles(std::string path = ".") {
    
    std::vector<std::string> files;
    
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (path.c_str())) != NULL)
    {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL)
        {
            string s = ent->d_name;
            s.erase(std::remove_if(s.begin(), s.end(), my_predicate), s.end());

            if(  !(s == "." || s == ".." || s == ".DS_Store") )
                files.push_back(ent->d_name);
        }
        closedir (dir);
    }
    else
    {
        /* could not open directory */
        perror ("");
    }
    
    return files;
}

std::vector<std::string> vocab;

void build_vocab_vector (const char * docs){
    static char buf[MAX_STRING] = "";
    FILE *file;
    file = fopen(docs, "r+");
    while(fgets(buf, MAX_STRING, file)){
        bool flag = false;
        for(int i = 0; i < vocab.size(); i++){
            string s = buf;
            s.erase(std::remove_if(s.begin(), s.end(), my_predicate), s.end());
            flag = false || (vocab[i] == s);
            if(flag)
                break;
        }
        if(!flag){
            string s = buf;
            s.erase(std::remove_if(s.begin(), s.end(), my_predicate), s.end());
            vocab.push_back(s);
        }
    }
    fclose(file);
    
}

int main(){
    
    std::vector<std::string> docs, queries;
    
    docs = getfiles(DOCS_PATH); // or pass which dir to open
    queries = getfiles(QUERY_PATH);
    
    for (int i = 0; i < docs.size(); ++i) {
        build_vocab_vector((string(DOCS_PATH) + string("/") + docs[i]).c_str());
    }
    
    cout<< "Size of vocab vector: "<<vocab.size()<<endl;
    
    cout<< "Computing the Vocab-Docs Matrix...."<<endl;
    short mat_d[2869][500];
    
    for(int i=0; i<vocab.size(); i++)    //This loops on the rows.
    {
        for(int j=0; j<docs.size(); j++) //This loops on the columns
        {
            mat_d[i][j] = 0;
        }
    }
    
    
    char buf[MAX_STRING];
    for(int i = 0; i < docs.size(); ++i){
        FILE *f = fopen((string(DOCS_PATH) + string("/") + docs[i]).c_str(), "r");
        while( fgets(buf, MAX_STRING, f) ){
            string s = buf;
            s.erase(std::remove_if(s.begin(), s.end(), my_predicate), s.end());
            for(int j = 0; j < vocab.size(); ++j){
                if( vocab[j] == s){
                    ++mat_d[j][i];
                }
            }
        }
    }
    
    cout<< "Computing the Vocab-Query Matrix...."<<endl;
    short mat_q[2869][5];
    
    for(int i=0; i<vocab.size(); i++)
    {
        for(int j=0; j<queries.size(); j++)
        {
            mat_q[i][j] = 0;
        }
    }
    
    for(int i = 0; i < queries.size(); ++i){
        FILE *f = fopen((string(QUERY_PATH) + string("/") + queries[i]).c_str(), "r");
        while( fgets(buf, MAX_STRING, f) ){
            string s = buf;
            s.erase(std::remove_if(s.begin(), s.end(), my_predicate), s.end());
            for(int j = 0; j < vocab.size(); ++j){
                if( vocab[j] == s){
                    ++mat_q[j][i];
                }
            }
        }
    }
// mat_Q is VALID
    
    cout << "Computing the Dot-Scalar Similarity and Cosine-Similarity..."<<endl;
    
    int result_dot[500];
    float result_cosine[500];
    float sum = 0;
    int Di = 0;
    int Qi = 0;
    
    for(int i=0; i<5; i++)
    {
        int top_ten[10];
        for(int j=0; j<500; j++)
        {
            sum = 0;
            Di = 0;
            Qi = 0;
            for(int k = 0; k < vocab.size(); ++k){
                sum = sum + mat_q[k][i] * mat_d[k][j];
                Di = Di + mat_d[k][j] * mat_d[k][j];
                Qi = Qi + mat_q[k][i] * mat_q[k][i];
            }
            result_dot[j] = sum;
            result_cosine[j] = sum/(sqrt(Di) * sqrt(Qi));
        }
        
        cout << "For Dot Product Similarity: "<<endl;
        for( int k = 0; k < 10; k++){
            int index = 0;
            int temp = result_dot[0];
            for(int j = 1; j<500; ++j){
                if(temp <= result_dot[j]){
                    index = j;
                    temp = result_dot[j];
                }
            }
            result_dot[index] = FOUND;
            top_ten[k] = index;
        }
        cout << "For Query "<<i+1<<" The top 10 files are: "<<endl;
        for(int k: top_ten)
            cout << docs[k].c_str() <<endl;
        
        cout << "For Cosine Similarity: "<<endl;
        for( int k = 0; k < 10; k++){
            int index = 0;
            float temp = result_cosine[0];
            for(int j = 1; j<500; ++j){
                if(temp <= result_cosine[j]){
                    index = j;
                    temp = result_cosine[j];
                }
            }
            result_cosine[index] = FOUND;
            top_ten[k] = index;
        }
        cout << "For Query "<<i+1<<" The top 10 files are: "<<endl;
        for(int k: top_ten)
            cout << docs[k].c_str() <<endl;
    }
    
    cout << "Performing LSI...."<<endl;
    cout << "Computing the SVD for vocab-doc matrix..."<<endl;
    
    MatrixXf A(2869, 500);
    for(int i=0; i<vocab.size(); i++)    //This loops on the rows.
    {
        for(int j=0; j<docs.size(); j++) //This loops on the columns
        {
            A(i, j) = mat_d[i][j];
        }
    }
    //JacobiSVD<MatrixXf> s(A, ComputeFullU | ComputeFullV);
    JacobiSVD<MatrixXf> svd = A.jacobiSvd(ComputeFullU | ComputeFullV);
    //  Diagonals are ordered by magnitude, therefore top - 5 signular values are top 5 diagnal values.
    MatrixXf U = svd.matrixU();
    MatrixXf V = svd.matrixV();
    MatrixXf d = svd.singularValues();
    MatrixXf S(5,5);
    for(int i=0; i<5; i++)    //This loops on the rows.
    {
        for(int j=0; j<5; j++) //This loops on the columns
        {
            S(i, j) = 0;
        }
    
    }
    
    for (int i = 0; i < 5; i ++){
        S(i, i) = d(i);
    }
    cout << "Top 5 singular vlaues are: "<<endl;
    for(int i=0; i<5; i++)    //This loops on the rows.
    {
        for(int j=0; j<5; j++) //This loops on the columns
        {
            cout << S(i, j) << "    ";
        }
        cout << "......."<<endl;
    }
    for(int i=0; i < 5; i++){
        int top_ten[10];
        for( int k = 0; k < 10; k++){
            int index = 0;
            float temp = U(0, i);
            for(int j = 1; j<2869; ++j){
                if(temp <= U(j, i)){
                    index = j;
                    temp = U(j, i);
                }
            }
            U(index, i) = FOUND;
            top_ten[k] = index;
        }
        cout << "For Concept "<<i+1<<" The top 10 words are: "<<endl;
        for(int k: top_ten)
            cout << vocab[k].c_str() <<endl;
        
        for( int k = 0; k < 10; k++){
            int index = 0;
            float temp = V(0, i);
            for(int j = 1; j<500; ++j){
                if(temp <= V(j, i)){
                    index = j;
                    temp = V(j, i);
                }
            }
            V(index, i) = FOUND;
            top_ten[k] = index;
        }
        cout << "For Concept "<<i+1<<" The top 10 files are: "<<endl;
        for(int k: top_ten)
            cout << docs[k].c_str() <<endl;
    }

    
    return 0;
}
