#ifndef CONTINUOUS_NON_LINEAR_OPTIMIZATION_READ_FILES_H
#define CONTINUOUS_NON_LINEAR_OPTIMIZATION_READ_FILES_H

#include <sstream>
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

namespace solver {

    enum cec2013_type_file {CEC2013_SHIFT_VECTOR, CEC2013_SHIFT_MATRIX, CEC2013_PERM_VECTOR,
                CEC2013_WEIGHT_VECTOR, CEC2013_SUB_COMPONENTS_VECTOR, CEC2013_ROTATION_MATRIX_25,
            CEC2013_ROTATION_MATRIX_50, CEC2013_ROTATION_MATRIX_100};

    class cec_read_files{
    public:
        string data_dir = "cec2013_cdatafiles";
        cec_read_files(){};

        template<typename vector_t>
        void read_vector_files(vector_t &x, size_t id_function, cec2013_type_file type){
            switch(type){
                case cec2013_type_file::CEC2013_SHIFT_VECTOR: {
                    size_t c = 0;
                    stringstream ss;
                    ss << data_dir << "/" << "F" << id_function << "-xopt.txt";
                    ifstream file(ss.str());
                    string value;
                    string line;
                    if (file.is_open()) {
                        stringstream iss;
                        while (getline(file, line)) {
                            iss << line;
                            while (getline(iss, value, ',')) {
                                x[c++] = stod(value);
                            }
                            iss.clear();
                        }
                        file.close();
                    } else {
                        cout << "Cannot open the datafiles '" << ss.str() << "'" << endl;
                    }
                    break;
                }
                case cec2013_type_file::CEC2013_PERM_VECTOR: {
                    stringstream ss;
                    ss << data_dir << "/" << "F" << id_function << "-p.txt";
                    ifstream file(ss.str());
                    size_t c = 0;
                    string value;
                    if (file.is_open()) {
                        while (getline(file, value, ',')) {
                            x[c++] = stod(value) - 1;
                        }
                        file.close();
                    } else {
                        cout << "Cannot open the datafiles '" << ss.str() << "'" << endl;
                    }
                    break;
                }
                case cec2013_type_file::CEC2013_WEIGHT_VECTOR: {
                    stringstream ss;
                    ss << data_dir << "/" << "F" << id_function << "-w.txt";
                    ifstream file(ss.str());
                    size_t c = 0;
                    string value;
                    if (file.is_open()) {
                        while (getline(file, value)) {
                            x[c++] = stod(value);
                        }
                    } else {
                        cout << "Cannot open the datafiles '" << ss.str() << "'" << endl;
                    }
                    break;
                }
                case cec2013_type_file::CEC2013_SUB_COMPONENTS_VECTOR: {
                    stringstream ss;
                    ss << data_dir << "/" << "F" << id_function << "-s.txt";
                    ifstream file(ss.str());
                    size_t c = 0;
                    string value;
                    if (file.is_open()) {
                        while (getline(file, value)) {
                            x[c++] = stod(value);
                        }
                    } else {
                        cout << "Cannot open the datafiles '" << ss.str() << "'" << endl;
                    }
                    break;
                }
                default:
                    throw std::logic_error("Invalid type file.");
            }
        }

        template<typename matrix_type>
        void read_matrix_files(matrix_type &x, size_t id_function, cec2013_type_file type){
            size_t sub_dim = 0;

            bool rotation_matrix = true;

            switch(type){
                case cec2013_type_file::CEC2013_ROTATION_MATRIX_25: {
                    sub_dim = 25;
                    break;
                }
                case cec2013_type_file::CEC2013_ROTATION_MATRIX_50: {
                    sub_dim = 50;
                    break;
                }
                case cec2013_type_file::CEC2013_ROTATION_MATRIX_100: {
                    sub_dim = 100;
                    break;
                }
                case cec2013_type_file::CEC2013_SHIFT_MATRIX: {
                    rotation_matrix = false;
                    break;
                }
                default:
                    throw std::logic_error("Invalid type file.");
            }

            if(rotation_matrix) {
                stringstream ss;
                ss << data_dir << "/" << "F" << id_function << "-R" << sub_dim << ".txt";

                ifstream file(ss.str());
                string value;
                string line;
                size_t i = 0;
                size_t j;

                if (file.is_open()) {
                    stringstream iss;
                    while (getline(file, line)) {
                        j = 0;
                        iss << line;
                        while (getline(iss, value, ',')) {
                            x(i, j) = stod(value);
                            j++;
                        }
                        iss.clear();
                        i++;
                    }
                    file.close();
                } else {
                    cout << "Cannot open the datafiles '" << ss.str() << "'" << endl;
                }
            }
            else{
                stringstream ss;
                ss<< data_dir <<"/" << "F" << id_function << "-xopt.txt";
                ifstream file (ss.str());
                string value;
                string line;
                int s[20] = {50, 50, 25, 25, 100, 100, 25, 25, 50, 25, 100, 25, 100, 50, 25, 25, 25, 100, 50, 25};
                if (file.is_open()){
                    stringstream iss;
                    size_t i = 0;
                    size_t i_sub_component = 0;

                    while ( getline(file, line) ){
                        if(i == s[i_sub_component]){
                            i = 0;
                            i_sub_component++;
                        }
                        iss<<line;
                        while (getline(iss, value, ',')){
                            x(i_sub_component,i) = stod(value);
                            i++;
                        }
                        iss.clear();
                    }
                    file.close();
                }
                else{
                    cout<<"Cannot open the OvectorVec datafiles '" <<ss.str() <<"'" <<endl;
                }
            }
        }
    };

}
#endif //CONTINUOUS_NON_LINEAR_OPTIMIZATION_READ_FILES_H
