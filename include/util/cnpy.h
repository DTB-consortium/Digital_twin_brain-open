//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <zlib.h>
#include <map>
#include <memory>
#include <cstdint>
#include <numeric>

using namespace std;

namespace cnpy {

    struct NpyArray {
        NpyArray(const vector<size_t>& _shape, size_t _word_size, bool _fortran_order) :
            shape(_shape), word_size(_word_size), fortran_order(_fortran_order)
        {
            num_vals = 1;
            for(size_t i = 0;i < shape.size();i++) num_vals *= shape[i];
            data_holder = shared_ptr<vector<char>>(
                new vector<char>(num_vals * word_size));
        }

        NpyArray() : shape(0), word_size(0), fortran_order(0), num_vals(0) { }

        template<typename T>
        T* data() {
            return reinterpret_cast<T*>(&(*data_holder)[0]);
        }

        template<typename T>
        vector<T> as_vec() {
            const T* p = data<T>();
            return vector<T>(p, p+num_vals);
        }

        size_t num_bytes() {
            return data_holder->size();
        }

        shared_ptr<vector<char>> data_holder;
        vector<size_t> shape;
        size_t word_size;
        bool fortran_order;
        size_t num_vals;
    };
   
    using npz_t = map<string, NpyArray>; 

    char BigEndianTest();
    char map_type(const type_info& t);
    template<typename T> vector<char> create_npy_header(const vector<size_t>& shape);
    void parse_npy_header(FILE* fp,size_t& word_size, vector<size_t>& shape, bool& fortran_order);
    void parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset);
    npz_t npz_load(string fname);
    NpyArray npz_load(string fname, string varname);
    NpyArray npy_load(string fname);

	bool npz_find(string fname, string varname);

    template<typename T> vector<char>& operator+=(vector<char>& lhs, const T rhs) {
        //write in little endian
        for(size_t byte = 0; byte < sizeof(T); byte++) {
            char val = *((char*)&rhs+byte); 
            lhs.push_back(val);
        }
        return lhs;
    }

    template<> vector<char>& operator+=(vector<char>& lhs, const string rhs); 
    template<> vector<char>& operator+=(vector<char>& lhs, const char* rhs); 


    template<typename T> void npy_save(string fname, const T* data, const vector<size_t> shape, string mode = "w") {
        FILE* fp = NULL;
        vector<size_t> true_data_shape; //if appending, the shape of existing + new data

		if (mode == "a") {
			fp = fopen(fname.c_str(), "r+b");
		}

        if(fp) {
            //file exists. we need to append to it. read the header, modify the array size
            size_t word_size;
            bool fortran_order;
            parse_npy_header(fp,word_size,true_data_shape,fortran_order);
            assert(!fortran_order);

            if(word_size != sizeof(T)) {
                cout<<"libnpy error: "<<fname<<" has word size "<<word_size<<" but npy_save appending data sized "<<sizeof(T)<<"\n";
                assert( word_size == sizeof(T) );
            }
            if(true_data_shape.size() != shape.size()) {
                cout<<"libnpy error: npy_save attempting to append misdimensioned data to "<<fname<<"\n";
                assert(true_data_shape.size() != shape.size());
            }

            for(size_t i = 1; i < shape.size(); i++) {
                if(shape[i] != true_data_shape[i]) {
                    cout<<"libnpy error: npy_save attempting to append misshaped data to "<<fname<<"\n";
                    assert(shape[i] == true_data_shape[i]);
                }
            }
            true_data_shape[0] += shape[0];
        }
        else {
			fp = fopen(fname.c_str(),"wb");
            true_data_shape = shape;
        }

        vector<char> header = create_npy_header<T>(true_data_shape);
        size_t nels = accumulate(shape.begin(),shape.end(),1,multiplies<size_t>());

        fseek(fp,0,SEEK_SET);
        fwrite(&header[0],sizeof(char),header.size(),fp);
        fseek(fp,0,SEEK_END);
        fwrite(data,sizeof(T),nels,fp);
        fclose(fp);
    }

    template<typename T> void npz_save(string zipname, string fname, const T* data, const vector<size_t>& shape, string mode = "w")
    {
        //first, append a .npy to the fname
        fname += ".npy";

        //now, on with the show
        FILE* fp = NULL;
        uint16_t nrecs = 0;
        size_t global_header_offset = 0;
        vector<char> global_header;

		if (mode == "a") {
			fp = fopen(zipname.c_str(), "r+b");
		}

        if(fp) {
            //zip file exists. we need to add a new npy file to it.
            //first read the footer. this gives us the offset and size of the global header
            //then read and store the global header. 
            //below, we will write the the new data at the start of the global header then append the global header and footer below it
            size_t global_header_size;
            parse_zip_footer(fp,nrecs,global_header_size,global_header_offset);
            fseek(fp,global_header_offset,SEEK_SET);
            global_header.resize(global_header_size);
            size_t res = fread(&global_header[0],sizeof(char),global_header_size,fp);
            if(res != global_header_size){
                throw runtime_error("npz_save: header read error while adding to existing zip");
            }
            fseek(fp,global_header_offset,SEEK_SET);
        }
        else {
            fp = fopen(zipname.c_str(),"wb");
        }

        vector<char> npy_header = create_npy_header<T>(shape);

        size_t nels = accumulate(shape.begin(),shape.end(),1,multiplies<size_t>());
        int nbytes = nels*sizeof(T) + npy_header.size();

        //get the CRC of the data to be added
        uint32_t crc = crc32(0L,(uint8_t*)&npy_header[0],npy_header.size());
        crc = crc32(crc,(uint8_t*)data,nels*sizeof(T));

        //build the local header
        vector<char> local_header;
        local_header += "PK"; //first part of sig
        local_header += (uint16_t) 0x0403; //second part of sig
        local_header += (uint16_t) 20; //min version to extract
        local_header += (uint16_t) 0; //general purpose bit flag
        local_header += (uint16_t) 0; //compression method
        local_header += (uint16_t) 0; //file last mod time
        local_header += (uint16_t) 0;     //file last mod date
        local_header += (uint32_t) crc; //crc
        local_header += (uint32_t) nbytes; //compressed size
        local_header += (uint32_t) nbytes; //uncompressed size
        local_header += (uint16_t) fname.size(); //fname length
        local_header += (uint16_t) 0; //extra field length
        local_header += fname;

        //build global header
        global_header += "PK"; //first part of sig
        global_header += (uint16_t) 0x0201; //second part of sig
        global_header += (uint16_t) 20; //version made by
        global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
        global_header += (uint16_t) 0; //file comment length
        global_header += (uint16_t) 0; //disk number where file starts
        global_header += (uint16_t) 0; //internal file attributes
        global_header += (uint32_t) 0; //external file attributes
        global_header += (uint32_t) global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
        global_header += fname;

        //build footer
        vector<char> footer;
        footer += "PK"; //first part of sig
        footer += (uint16_t) 0x0605; //second part of sig
        footer += (uint16_t) 0; //number of this disk
        footer += (uint16_t) 0; //disk where footer starts
        footer += (uint16_t) (nrecs+1); //number of records on this disk
        footer += (uint16_t) (nrecs+1); //total number of records
        footer += (uint32_t) global_header.size(); //nbytes of global headers
        footer += (uint32_t) (global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
        footer += (uint16_t) 0; //zip file comment length

        //write everything      
        fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
        fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);
        fwrite(data,sizeof(T),nels,fp);
        fwrite(&global_header[0],sizeof(char),global_header.size(),fp);
        fwrite(&footer[0],sizeof(char),footer.size(),fp);
        fclose(fp);
    }

    template<typename T> void npy_save(string fname, const vector<T> data, string mode = "w") {
        vector<size_t> shape;
        shape.push_back(data.size());
        npy_save(fname, &data[0], shape, mode);
    }

    template<typename T> void npz_save(string zipname, string fname, const vector<T> data, string mode = "w") {
        vector<size_t> shape;
        shape.push_back(data.size());
        npz_save(zipname, fname, &data[0], shape, mode);
    }

    template<typename T> vector<char> create_npy_header(const vector<size_t>& shape) {  

        vector<char> dict;
        dict += "{'descr': '";
        dict += BigEndianTest();
        dict += map_type(typeid(T));
        dict += to_string(sizeof(T));
        dict += "', 'fortran_order': False, 'shape': (";
        dict += to_string(shape[0]);
        for(size_t i = 1;i < shape.size();i++) {
            dict += ", ";
            dict += to_string(shape[i]);
        }
        if(shape.size() == 1) dict += ",";
        dict += "), }";
        //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
        int remainder = 16 - (10 + dict.size()) % 16;
        dict.insert(dict.end(),remainder,' ');
        dict.back() = '\n';

        vector<char> header;
        header += (char) 0x93;
        header += "NUMPY";
        header += (char) 0x01; //major version of numpy format
        header += (char) 0x00; //minor version of numpy format
        header += (uint16_t) dict.size();
        header.insert(header.end(),dict.begin(),dict.end());

        return header;
    }


}

#endif
