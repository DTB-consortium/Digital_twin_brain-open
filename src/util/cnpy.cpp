//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#include "util/cnpy.h"
#include <complex>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iomanip>

char cnpy::BigEndianTest() {
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
}

char cnpy::map_type(const type_info& t)
{
    if(t == typeid(float) ) return 'f';
    if(t == typeid(double) ) return 'f';
    if(t == typeid(long double) ) return 'f';

    if(t == typeid(int) ) return 'i';
    if(t == typeid(char) ) return 'i';
    if(t == typeid(short) ) return 'i';
    if(t == typeid(long) ) return 'i';
    if(t == typeid(long long) ) return 'i';

    if(t == typeid(unsigned char) ) return 'u';
    if(t == typeid(unsigned short) ) return 'u';
    if(t == typeid(unsigned long) ) return 'u';
    if(t == typeid(unsigned long long) ) return 'u';
    if(t == typeid(unsigned int) ) return 'u';

    if(t == typeid(bool) ) return 'b';

    if(t == typeid(complex<float>) ) return 'c';
    if(t == typeid(complex<double>) ) return 'c';
    if(t == typeid(complex<long double>) ) return 'c';

    else return '?';
}

template<> vector<char>& cnpy::operator+=(vector<char>& lhs, const string rhs) {
    lhs.insert(lhs.end(),rhs.begin(),rhs.end());
    return lhs;
}

template<> vector<char>& cnpy::operator+=(vector<char>& lhs, const char* rhs) {
    //write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for(size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

void cnpy::parse_npy_header(FILE* fp, size_t& word_size, vector<size_t>& shape, bool& fortran_order) {  
    char buffer[256];
    size_t res = fread(buffer,sizeof(char),11,fp);       
    if(res != 11)
        throw runtime_error("parse_npy_header: failed fread");
    string header = fgets(buffer,256,fp);
    assert(header[header.size()-1] == '\n');

    int loc1, loc2;

    //fortran order
    loc1 = header.find("fortran_order")+16;
    fortran_order = (header.substr(loc1,4) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    string str_shape = header.substr(loc1+1,loc2-loc1-1);
    size_t ndims;
    if(str_shape[str_shape.size()-1] == ',') ndims = 1;
    else ndims = count(str_shape.begin(),str_shape.end(),',')+1;
    shape.resize(ndims);
    for(size_t i = 0;i < ndims;i++) {
        loc1 = str_shape.find(",");
        shape[i] = stoul(str_shape.substr(0,loc1));
		//cout << shape[i] << endl;
        str_shape = str_shape.substr(loc1+1);
    }

    //endian, word size, data type
    //byte order code | stands for not applicable. 
    //not sure when this applies except for byte array
    loc1 = header.find("descr")+9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    //char type = header[loc1+1];
    //assert(type == map_type(T));

    string str_ws = header.substr(loc1+2);
    loc2 = str_ws.find("'");
    word_size = stoul(str_ws.substr(0,loc2));
	//cout << word_size << endl;
}

void cnpy::parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset)
{
    vector<char> footer(22);
    fseek(fp,-22,SEEK_END);
    size_t res = fread(&footer[0],sizeof(char),22,fp);
    if(res != 22)
        throw runtime_error("parse_zip_footer: failed fread");

    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no = *(uint16_t*) &footer[4];
    disk_start = *(uint16_t*) &footer[6];
    nrecs_on_disk = *(uint16_t*) &footer[8];
    nrecs = *(uint16_t*) &footer[10];
    global_header_size = *(uint32_t*) &footer[12];
    global_header_offset = *(uint32_t*) &footer[16];
    comment_len = *(uint16_t*) &footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
}

cnpy::NpyArray load_the_npy_file(FILE* fp) {
    vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(fp,word_size,shape,fortran_order);

    cnpy::NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(),1,arr.num_bytes(),fp);
    if(nread != arr.num_bytes())
        throw runtime_error("load_the_npy_file: failed fread");
    return arr;
}

void skip_the_npy_file(FILE* fp) {
    vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(fp,word_size,shape,fortran_order);

    size_t num_vals = 1;
	for(size_t i = 0;i < shape.size();i++)
		num_vals *= shape[i];
	fseek(fp,num_vals * word_size,SEEK_CUR);
}


cnpy::npz_t cnpy::npz_load(string fname) {
	FILE* fp;
	fp = fopen(fname.c_str(), "rb");
	if (!fp) {
		printf("npz_load: Error! Unable to open file %s!\n", fname.c_str());
	}
    assert(fp);

    cnpy::npz_t arrays;  

    while(1) {
        vector<char> local_header(30);
        size_t headerres = fread(&local_header[0],sizeof(char),30,fp);
        if(headerres != 30)
            throw runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        string varname(name_len,' ');
        size_t vname_res = fread(&varname[0],sizeof(char),name_len,fp);
        if(vname_res != name_len)
            throw runtime_error("npz_load: failed fread");

        //erase the lagging .npy        
        varname.erase(varname.end()-4,varname.end());

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        if(extra_field_len > 0) {
            vector<char> buff(extra_field_len);
            size_t efield_res = fread(&buff[0],sizeof(char),extra_field_len,fp);
            if(efield_res != extra_field_len)
                throw runtime_error("npz_load: failed fread");
        }

        arrays[varname] = load_the_npy_file(fp);
    }

    fclose(fp);
    return arrays;  
}

bool cnpy::npz_find(string fname, string varname)
{
	FILE* fp;
	fp = fopen(fname.c_str(), "rb");
    if(!fp) {
        printf("npz_load: Error! Unable to open file %s!\n",fname.c_str());
        return false;
    }       

    while(1) {
        vector<char> local_header(30);
        size_t header_res = fread(&local_header[0],sizeof(char),30,fp);
        if(header_res != 30)
        {
        	break;
        }

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        string vname(name_len,' ');
        size_t vname_res = fread(&vname[0],sizeof(char),name_len,fp);      
        if(vname_res != name_len)
            throw runtime_error("npz_load: failed fread");
        vname.erase(vname.end()-4,vname.end()); //erase the lagging .npy

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        fseek(fp,extra_field_len,SEEK_CUR); //skip past the extra field
		
        if(vname == varname) {
            fclose(fp);
            return true;
        }
        else {
            //skip past the data
          skip_the_npy_file(fp);
        }
    }

    fclose(fp);
	return false;
}

cnpy::NpyArray cnpy::npz_load(string fname, string varname) {
	FILE* fp;
	fp = fopen(fname.c_str(), "rb");
    if(!fp) {
        printf("npz_load: Error! Unable to open file %s!\n",fname.c_str());
        abort();
    }       

    while(1) {
        vector<char> local_header(30);
        size_t header_res = fread(&local_header[0],sizeof(char),30,fp);
        if(header_res != 30)
            throw runtime_error("npz_load: failed fread");

        //if we've reached the global header, stop reading
        if(local_header[2] != 0x03 || local_header[3] != 0x04) break;

        //read in the variable name
        uint16_t name_len = *(uint16_t*) &local_header[26];
        string vname(name_len,' ');
        size_t vname_res = fread(&vname[0],sizeof(char),name_len,fp);      
        if(vname_res != name_len)
            throw runtime_error("npz_load: failed fread");
        vname.erase(vname.end()-4,vname.end()); //erase the lagging .npy

        //read in the extra field
        uint16_t extra_field_len = *(uint16_t*) &local_header[28];
        fseek(fp,extra_field_len,SEEK_CUR); //skip past the extra field

        if(vname == varname) {
            NpyArray array = load_the_npy_file(fp);
            fclose(fp);
            return array;
        }
        else {
            //skip past the data
            skip_the_npy_file(fp);
        }
    }

    fclose(fp);
    printf("npz_load: Error! Variable name %s not found in %s!\n",varname.c_str(),fname.c_str());
    abort();
}

cnpy::NpyArray cnpy::npy_load(string fname) {
	FILE*fp;
    fp = fopen(fname.c_str(), "rb");

    if(!fp) {
        printf("npy_load: Error! Unable to open file %s!\n",fname.c_str());
        abort();  
    }

    NpyArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}



