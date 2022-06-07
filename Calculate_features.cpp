#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
using namespace std;

// Constant define
#define nAtom 192
#define nO 64
#define nH 128
#define rCut 6              // ans
#define Oxyz_threshold 16   
#define Hxyz_threshold 32   
// Matrix define
typedef vector < double > VEC;
typedef vector < vector < double >> MAT2;
typedef vector < vector < vector < double >>> MAT3;
typedef vector < vector < vector < vector < double >>>> MAT4;

typedef vector < int > VECI;
typedef vector < vector < int >> MATI2;

// Function Statement
    ostream& operator<<(ostream& out, const MAT2& m);   // Checked
    ostream& operator<<(ostream& out, const VEC& v);    // Checked
    VEC     operator+(const VEC& v1, const VEC& v2);    // Checked
    VEC     operator-(const VEC& v1, const VEC& v2);    // Checked
    MAT2    operator+(const MAT2& m1, const MAT2& m2);  // Checked  
    MAT2    operator-(const MAT2& m1, const MAT2& m2);  // Checked
    double  operator*(const VEC& v1, const VEC& v2);    // Checked
    VEC     operator*(const VEC& v, const double k);    // Checked    
    VEC     operator*(const double k, const VEC& v);    // Checked
    VEC     operator*(const MAT2& mat, const VEC& v);   // Checked
    VEC     operator*(const VEC& v, const MAT2& mat);   // Checked
    double norm(VEC& v);                                // Checked 
    VEC base(VEC& v);                                   // Checked
    VEC CrossProduct_3D(VEC& v1, VEC& v2);              // Checked
    MAT2 Transpose(MAT2&);                              // Checked
    VEC PBC(VEC&, VEC&);                                // Checked
    void Load_Box(VEC&, string, int);                   // Checked
    void Load_Coord(MAT2&, string, int, int);           // Checked
    void Merge_coords(MAT2&, MAT2&, MAT2&);             // Checked
    void Calculate_rij(MAT2&, MAT3&, MAT2&, VEC&);      // Checked
    vector<int> argsort(const VEC& array);              // Checked
    void Local_frame(MAT3&, MATI2&, MAT3&);             // Checked 
    void Sort_r(MATI2&, MAT2&); 
    void Calculate_discriptor(MAT2&, MAT4&, MAT2&, MAT3&, MAT3&, MATI2&, int, string);
    
// main
    int main(){
        string config_path = "./";
         
        VEC box(3);
        MAT2 Oxyz(nO, VEC(3));
        MAT2 Hxyz(nH, VEC(3));
        MAT2 coords(nAtom, VEC(3));
        MAT2 norm_r(nAtom, VEC(nAtom));
        MAT3 vec_r(nAtom, MAT2(nAtom, VEC(3)));
        MATI2 sort_index(nAtom, VECI(nAtom));
        MAT3 total_frame(nAtom, MAT2(3, VEC(3)));
       
        MAT2 features_O(nO, VEC(4 * (Oxyz_threshold + Hxyz_threshold)));
        MAT2 features_H(nH, VEC(4 * (Oxyz_threshold + Hxyz_threshold)));

        MAT4 dfeatures_O(nO, MAT3(4 * (Oxyz_threshold + Hxyz_threshold), MAT2(nAtom, VEC(3))));
        MAT4 dfeatures_H(nH, MAT3(4 * (Oxyz_threshold + Hxyz_threshold), MAT2(nAtom, VEC(3))));

        Load_Box(box, config_path + "Box.txt", 3);
        Load_Coord(Oxyz, config_path + "Oxyz.txt", nO, 3);
        Load_Coord(Hxyz, config_path + "Hxyz.txt", nH, 3);
        Merge_coords(coords, Oxyz, Hxyz);

        Calculate_rij(norm_r, vec_r, coords, box);
        Sort_r(sort_index, norm_r); 
         
        Local_frame(total_frame, sort_index, vec_r);

        Calculate_discriptor(features_O, dfeatures_O, norm_r, vec_r, total_frame, sort_index, 0, "O");
        Calculate_discriptor(features_H, dfeatures_H, norm_r, vec_r, total_frame, sort_index, 1, "H");
        cout << total_frame.size() << endl; 
        cout << total_frame[0] << endl;
    
    }
// Matrix operator
    ostream& operator<<(ostream& out, const MAT2& m){
        // Print 2D matrix to screen
        for (int x = 0; x < m.size(); x++){
            for (int y = 0; y < m[0].size(); y++){
                out << m[x][y] << "  ";
            }
            out << endl;
        }
        return(out);
    }
    ostream& operator<<(ostream& out, const VEC& v){
        // Print vector to srceen
        for (int x = 0; x < v.size(); x++){
            out << v[x] << "  ";
        }
        return(out);
    }
    VEC operator+(const VEC& v1, const VEC& v2){
        // Vector add up
        if(v1.size() != v2.size()){
            cout << "Vector shape dosen't match in vector add !" << endl;
            exit(-1);
        }
        VEC result(v1.size());
        for (int index = 0; index < v1.size(); index++){
            result[index] = v1[index] + v2[index];
        }
        return(result);
    }
    VEC operator-(const VEC& v1, const VEC& v2){
        if(v1.size() != v2.size()){
            cout << "Vector shape dosen't match in vector sub !" << endl;
            exit(-1);
        }
        VEC result(v1.size());
        for (int index = 0; index < v1.size(); index++){
            result[index] = v1[index] - v2[index];
        }
        return(result);
    }
    MAT2 operator+(const MAT2& m1, const MAT2& m2){
        if(m1.size() != m2.size() || m1[0].size() != m2[0].size()){
            cout << "Shape dosen't match in matrix add up" << endl;
            exit(-1);
        }
        MAT2 result(m1.size(),VEC(m1[0].size())); 
        for (int i = 0; i < m1.size(); i++){
            for (int j = 0; j < m1[0].size(); j++){
                result[i][j] = m1[i][j] + m2[i][j];
            }
        }
        return(result);
    } 
    MAT2 operator-(const MAT2& m1, const MAT2& m2){
        if(m1.size() != m2.size() || m1[0].size() != m2[0].size()){
            cout << "Shape dosen't match in matrix add up" << endl;
            exit(-1);
        }
        MAT2 result(m1.size(),VEC(m1[0].size())); 
        for (int i = 0; i < m1.size(); i++){
            for (int j = 0; j < m1[0].size(); j++){
                result[i][j] = m1[i][j] - m2[i][j];
            }
        }
        return(result);
    } 
    double operator*(const VEC& v1, const VEC& v2){
        // Vec dot mult
        if(v1.size() != v2.size()){
            cout << "Vector shape dosen't match in vector dot mult !" << endl;
            exit(-1);
        }
        double result = 0;
        for (int index = 0; index < v1.size(); index++) {
            result += v1[index] * v2[index];
        }
        return(result);
    }
    VEC operator*(const VEC& v, const double k){
        VEC result(v.size());
        for (int i = 0; i < v.size(); i++){
            result[i] = v[i] * k;
        }
        return(result);
    }     
    VEC operator*(const double k, const VEC& v){
        VEC result(v.size());
        for (int i = 0; i < v.size(); i++){
            result[i] = v[i] * k;
        }
        return(result);
    }     
    double norm(VEC& v){
        double result = 0;
        result = sqrt(v * v);
        return(result);
    }
    VEC base(VEC& v){
        VEC result(v.size());
        result = (1 / norm(v)) * v;
        return(result);
    }
    MAT2 Transpose(const MAT2& mat){
        MAT2 result(mat[0].size(), VEC(mat.size()));
        for (int i = 0; i < mat.size(); i++){
            for (int j = 0; j < mat[0].size(); j++){
                result[j][i] = mat[i][j];
            }
        }
        return(result); 
    }
    VEC CrossProduct_3D(VEC& v1, VEC& v2){
        VEC result = {0, 0, 0};
        result[0] = + (v1[1] * v2[2] - v2[1] * v1[2]);
        result[1] = - (v1[0] * v2[2] - v2[0] * v1[2]);
        result[2] = + (v1[0] * v2[1] - v2[0] * v1[1]);
        return(result);
    }
    VEC operator*(const MAT2& mat, const VEC& v){
        if (mat[0].size() != v.size()){
            cout << "[ ERROR ] : shape dosn't match in mat * vector !" << endl;
            exit(-1);
        }
        VEC result(mat.size());
        for (int index = 0; index < mat.size(); index++){
            result[index] = mat[index] * v;
        }
        return(result);
    }
    VEC operator*(const VEC& v, const MAT2& mat){
        if (mat.size() != v.size()){
            cout << "[ ERROR ] : shape dosn't match in vector * mat !" << endl;
            exit(-1);
        }
        MAT2 mat_t(mat[0].size(), VEC(mat.size()));
        mat_t = Transpose(mat);
        VEC result(mat[0].size());
        for (int index = 0; index < mat.size(); index++){
            result[index] = mat_t[index] * v;
        }
        return(result);
    }
// Data operator
    VEC PBC(VEC& coord, VEC& box){
        // Periodic Boundary Condition
        VEC result = {0, 0, 0};
        result[0] = coord[0] - round(coord[0] / box[0]) * box[0];
        result[1] = coord[1] - round(coord[1] / box[1]) * box[1];
        result[2] = coord[2] - round(coord[2] / box[2]) * box[2];
        return(result);
    }
    void Load_Box(VEC& box, string path, int shape){
        ifstream fp(path);
        for (int i = 0; i < shape; i++){
            fp >> box[i];
        }
        fp.close();
    }
    void Load_Coord(MAT2& mat, string path, int shape_x, int shape_y){
        ifstream fp(path);
        for (int i = 0; i < shape_x; i ++){
            for (int j = 0; j < shape_y; j++){
                fp >> mat[i][j];
            }
        }
        fp.close();
    }
    void Merge_coords(MAT2& mat, MAT2& Oxyz, MAT2& Hxyz){
        for (int d = 0; d < 3 ; d++){
            for (int i = 0; i < nO; i++){
                mat[i][d] = Oxyz[i][d];        
            }
            for (int i = 0; i < nH; i++){
                mat[i + nO][d] = Hxyz[i][d];        
            }
        }
    }
    void Calculate_rij(MAT2& r, MAT3& vec_r, MAT2& coords, VEC& Box){
        VEC vec_buffer = {0, 0, 0};
        for (int i = 0; i < coords.size(); i++){
            for (int j = 0; j < coords.size(); j++){
                vec_buffer = coords[i] - coords[j];
                vec_buffer = PBC(vec_buffer, Box);
                r[i][j] = norm(vec_buffer);
                vec_r[i][j] = vec_buffer;
            }
        }
    }
    vector<int> argsort(const vector<double>& array){   
        const int array_len(array.size());
        vector<int> array_index(array_len, 0);
        for (int i = 0; i < array_len; ++i)
            array_index[i] = i; 
            sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});
        return array_index;
    }
    void Local_frame(MAT3& total_frame, MATI2& sorted_index, MAT3& vec_r){
        VEC vec_a, vec_b, vec_buffer;
        MAT2 local_frame(3, VEC(3));
        for (int icenter = 0; icenter < vec_r.size(); icenter++){
            vec_a = vec_r[icenter][sorted_index[icenter][1]];
            vec_b = vec_r[icenter][sorted_index[icenter][2]];
            vec_buffer = CrossProduct_3D(vec_a, vec_b);
            if (norm(vec_buffer) == 0){
                vec_b = vec_r[icenter][sorted_index[icenter][3]];
            }
            local_frame[0] = base(vec_a);
            local_frame[1] = vec_b - (local_frame[0] * vec_b) * local_frame[0];
            local_frame[1] = base(local_frame[1]);
            local_frame[2] = CrossProduct_3D(local_frame[0], local_frame[1]);
            local_frame[2] = base(local_frame[2]);
            total_frame[icenter] = local_frame;
            if (icenter == 0){
                cout << local_frame << endl;
                cout << sorted_index[0][1] << "  " << sorted_index[0][2] << endl;
            }
            /* 
                cout << local_frame[0] * local_frame[1] << endl;
                cout << local_frame[0] * local_frame[2] << endl;
                cout << local_frame[1] * local_frame[2] << endl;
            */
        }
    }
    void Sort_r(MATI2& sort_index, MAT2& norm_r){
        for (int icenter = 0; icenter < norm_r.size(); icenter++){
            sort_index[icenter] = argsort(norm_r[icenter]);     
        }
    }
    void Calculate_discriptor(MAT2& features, MAT4& dfeatures, MAT2& norm_r, MAT3& vec_r, MAT3& total_frame, MATI2& sort_index, int center_type, string save_name){
        int center_range[] = {0, nO};
        if(center_type == 1){
            center_range[0] = nO;
            center_range[1] = nAtom;
        }
        
        int index_i, index_j;
        double rij, rij_2, rij_4;
        VEC vec_rij;    
   
        VEC dfeature_r = {0, 0, 0};
        VEC dfeature_1 = {0, 0, 0}; 
        VEC dfeature_2 = {0, 0, 0}; 
        VEC dfeature_3 = {0, 0, 0};
        
        int Oxyz_num, Hxyz_num;

        for (int i = center_range[0]; i < center_range[1]; i++){
            index_i = i;
            Oxyz_num = 0;
            Hxyz_num = 0;
            for (int j = 0; j < nAtom; j++){
                index_j = sort_index[i][j];
                rij = norm_r[i][index_j];
                if (rij > rCut | rij == 0){continue;}
                rij_2 = rij * rij;
                rij_4 = rij_2 * rij_2;
                vec_rij = total_frame[index_i] * vec_r[index_i][index_j];
                    
                dfeature_r =  ( - 1 / (rij * rij * rij) * vec_rij * total_frame[i]);
                
                dfeature_1 = {rij_2 - 2 * vec_rij[0] * vec_rij[0], - 2 * vec_rij[0] * vec_rij[1], - 2 * vec_rij[0] * vec_rij[2]};
                dfeature_2 = {- 2 * vec_rij[1] * vec_rij[0], rij_2 - 2 * vec_rij[1] * vec_rij[1], - 2 * vec_rij[1] * vec_rij[2]};
                dfeature_3 = {- 2 * vec_rij[2] * vec_rij[0], - 2 * vec_rij[2] * vec_rij[1], rij_2 - 2 * vec_rij[2] * vec_rij[2]};
                
                dfeature_1 = (1 / rij_4) * dfeature_1 * total_frame[i];
                dfeature_2 = (1 / rij_4) * dfeature_2 * total_frame[i];
                dfeature_3 = (1 / rij_4) * dfeature_3 * total_frame[i];
                if ( index_j < nO && Oxyz_num < Oxyz_threshold){
                    features[index_i - center_range[0]][Oxyz_num * 4    ] = 1 / rij;
                    features[index_i - center_range[0]][Oxyz_num * 4 + 1] = vec_rij[0] / rij_2;
                    features[index_i - center_range[0]][Oxyz_num * 4 + 2] = vec_rij[1] / rij_2;
                    features[index_i - center_range[0]][Oxyz_num * 4 + 3] = vec_rij[2] / rij_2;
                    
                    for (int d = 0; d < 3; d++){
                        dfeatures[index_i - center_range[0]][Oxyz_num * 4    ][index_i][d] += dfeature_r[d];
                        dfeatures[index_i - center_range[0]][Oxyz_num * 4 + 1][index_i][d] += dfeature_1[d];
                        dfeatures[index_i - center_range[0]][Oxyz_num * 4 + 2][index_i][d] += dfeature_2[d];
                        dfeatures[index_i - center_range[0]][Oxyz_num * 4 + 3][index_i][d] += dfeature_3[d];
                         
                        dfeatures[index_i - center_range[0]][Oxyz_num * 4    ][index_j][d] -= dfeature_r[d];
                        dfeatures[index_i - center_range[0]][Oxyz_num * 4 + 1][index_j][d] -= dfeature_1[d];
                        dfeatures[index_i - center_range[0]][Oxyz_num * 4 + 2][index_j][d] -= dfeature_2[d];
                        dfeatures[index_i - center_range[0]][Oxyz_num * 4 + 3][index_j][d] -= dfeature_3[d];
                    }
                    Oxyz_num += 1;
                } 
                if ( nO <= index_j && Hxyz_num < Hxyz_threshold){
                    features[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4    ] = 1 / rij;
                    features[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 1] = vec_rij[0] / rij_2;
                    features[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 2] = vec_rij[1] / rij_2;
                    features[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 3] = vec_rij[2] / rij_2;
                    
                    for (int d = 0; d < 3; d++){
                        dfeatures[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4    ][index_i][d] += dfeature_r[d];
                        dfeatures[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 1][index_i][d] += dfeature_1[d];
                        dfeatures[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 2][index_i][d] += dfeature_2[d];
                        dfeatures[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 3][index_i][d] += dfeature_3[d];
                         
                        dfeatures[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4    ][index_j][d] -= dfeature_r[d];
                        dfeatures[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 1][index_j][d] -= dfeature_1[d];
                        dfeatures[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 2][index_j][d] -= dfeature_2[d];
                        dfeatures[index_i - center_range[0]][4 * Oxyz_threshold + Hxyz_num * 4 + 3][index_j][d] -= dfeature_3[d];
                    }
                    Hxyz_num += 1;
                } 
            }
        }            
        ofstream feature_fp("./features/feature_" + save_name + ".txt");
        ofstream dfeature_fp("./features/feature_d" + save_name + ".txt");
        for (int i = center_range[0]; i < center_range[1]; i++){
            for (int k = 0; k < (Oxyz_threshold + Hxyz_threshold) * 4; k++){
                feature_fp << features[i - center_range[0]][k] << "  ";
            }
            feature_fp << endl;
        }
        feature_fp.close();
        for (int i = center_range[0]; i < center_range[1]; i++){
            for (int k = 0; k < (Oxyz_threshold + Hxyz_threshold) * 4; k++){
                for (int j = 0; j < nAtom; j++){
                    for (int d = 0; d < 3; d++){
                        dfeature_fp << dfeatures[i - center_range[0]][k][j][d] << "  ";
                    }
                }
            }
        }
        dfeature_fp.close();
    }

