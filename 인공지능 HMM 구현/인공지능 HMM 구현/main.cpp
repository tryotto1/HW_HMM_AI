#pragma once
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#define REF_FILE "reference.txt"
#define RST_FILE "recognized.txt"
#define unigram_address	"unigram.txt"
#define bigram_address	"bigram.txt"
#define dict_address	"dictionary.txt"
#define test_address	"tst/f/bf/826z358.txt"

#define max_test_len 10000
#define max_word_phone_num 6
#define MAX_BUF 80
#define INF 100000000000
#define max_word_cnt 13
#define max_phone_cnt 25
#define max_buff_len 1000

#include "hmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <cmath>

using namespace std;

vector<float> unigram;
vector<vector<float> > bigram;
vector<vector<int> > word2phone;
vector<vector<float> > testcase;

float coef_log[15][20][5] = { 0., };		// log 계산을 빨리 할 수 있도록 하기 위함
float total_prob[1000][13][20] = { -INF, };	// total_prob[time][word][phones in word]
pair<int, int> max_idx[1000][13][20];		// max_idx[time][word][phones in word] = {word, phone}

int lambda = 22;
_total_word_hmm total_word_hmm;

void fileNameRecog(const char* buf, char* result);
void lab2rec(char* str);

float log_fun(float x);
float get_pes(int time, int word, int state);

void to_next_state(int word, int cur_s, int next_s, int time);
void to_next_word(int word, int cur_s, int next_s, int time);
void viterbi(FILE* rst_file);

word_hmm make_word_hmm(int word_idx);
vector<float> make_unigram();
vector<vector<float> > make_bigram();
vector<vector<int> > dict_word2phone();
vector<vector<float> > make_testcase();

const char* int_to_str(int num);
const char* int_to_phone(int num);
int phone_to_int(char* s);
int word_to_int(char* s);
int phone_to_int(const char* s);
int word_to_int(const char* s);

void fileNameRecog(const char* buf, char* result) {
    while (*buf != '\0' && *buf != '\n') {
        if (*buf == '\"') {
            buf++;
            continue;
        }
        if (*buf == '/')
            *result = '\\';
        else
            *result = *buf;

        buf++;
        result++;
    }
    *(result - 3) = 't';
    *(result - 2) = 'x';
    *(result - 1) = 't';
    *result = '\0';
}

void lab2rec(char* str) {
    int len = strlen(str);
    str[len - 5] = 'r';
    str[len - 4] = 'e';
    str[len - 3] = 'c';
}

float log_fun(float x) {
	if (x == 0.0f)
		return -INF;
	else
		return log(x);
}

float get_pes(int time, int word, int state) {
	int t = word2phone[word][(state - 1) / 3];
	int phone_s_idx = (state - 1) % 3; 

	float tmp_sum[N_PDF] = { 0, };
	for (int i = 0; i < N_PDF; i++) {
		float exp_val = 0;		
		for (int j = 0; j < N_DIMENSION; j++) {
			exp_val += pow(testcase[time][j] - phones[t].state[phone_s_idx].pdf[i].mean[j], 2.0) / phones[t].state[phone_s_idx].pdf[i].var[j];
		}		
		exp_val *= (float)1.0 / 2.0;
		exp_val *= (-1);
		tmp_sum[i] += exp_val;

		if (coef_log[word][state][i] == 0){
			float tmp_coef =
				logf(phones[t].state[phone_s_idx].pdf[i].weight) - ((float)N_DIMENSION / 2.0) * logf(2.0 * M_PI);

			for (int j = 0; j < N_DIMENSION; j++) 
				tmp_coef -= logf(sqrt(phones[t].state[phone_s_idx].pdf[i].var[j]));

			coef_log[word][state][i] = tmp_coef;
		}				
		tmp_sum[i] += coef_log[word][state][i];
	}

	/* implementation issues */
	float rst = tmp_sum[0] + logf(1.0 + exp(tmp_sum[1] - tmp_sum[0]));

	return rst;
}

word_hmm make_word_hmm(int word_idx) {
	word_hmm tmp_word;

	// store all the phones in the word
	for (int i = 0; i < word2phone[word_idx].size(); i++) {
		if (word2phone[word_idx][i] == phone_to_int("sp")) {
			tmp_word.list_phone.push_back(word2phone[word_idx][i]);
		}
		else {
			for (int j = 0; j < 3; j++)
				tmp_word.list_phone.push_back(word2phone[word_idx][i]);
		}
	}

	// register name of the word
	tmp_word.word_name = int_to_str(word_idx);

	// how many states are in the word
	tmp_word.w_ns = 0;
	for (int i = 0; i < word2phone[word_idx].size(); i++) {
		int p_idx = word2phone[word_idx][i];

		if (p_idx != phone_to_int("sp"))	// "sp" 가 아닐 경우
			tmp_word.w_ns += 3;
		else				// "sp" 일경우
			tmp_word.w_ns += 1;
	}

	// make new state array of the word
	for (int i = 0; i < word2phone[word_idx].size(); i++) {
		int p_idx = word2phone[word_idx][i];

		if (p_idx != phone_to_int("sp")) {	// "sp" 가 아닐 경우
			for (int i = 0; i < 3; i++)
				tmp_word.w_state.push_back(phones[p_idx].state[i]);
		}
		else {				// "sp" 일경우
			for (int i = 0; i < 1; i++)
				tmp_word.w_state.push_back(phones[p_idx].state[i]);
		}
	}

	// make new transition matrix for word
	vector<vector<float> > tmp_tp(tmp_word.w_ns + 2, vector<float>(tmp_word.w_ns + 2, (float)0.0));
	int start_x = 1, start_y = 1;
	for (int i = 0; i < word2phone[word_idx].size(); i++) {
		int p_idx = word2phone[word_idx][i];

		if (p_idx != phone_to_int("sp")) {	// "sp" 가 아닐 경우			
			for (int j = 0; j < 3; j++)
				for (int k = 0; k < 4; k++)
					tmp_tp[start_y + j][start_x + k] = phones[p_idx].tp[1 + j][1 + k];
			start_x += 3;
			start_y += 3;
		}
		else { 				// "sp" 일경우
			for (int j = 0; j < 1; j++)
				for (int k = 0; k < 2; k++)
					tmp_tp[start_y + j][start_x + k] = phones[p_idx].tp[1 + j][1 + k];
			start_x += 1;
			start_y += 1;
		}
	}
	// the starting point needs additional calculation
	tmp_tp[0][1] = (float)1;

	// the phone "<sp>" needs additional calculation
	tmp_tp[tmp_word.w_ns - 1][tmp_word.w_ns + 1] = tmp_tp[tmp_word.w_ns - 1][tmp_word.w_ns] * phones[phone_to_int("sp")].tp[0][2];
	tmp_tp[tmp_word.w_ns - 1][tmp_word.w_ns] *= phones[phone_to_int("sp")].tp[0][1];

	// change prob to log
	for (int i = 0; i < tmp_word.w_ns + 2; i++) {
		for (int j = 0; j < tmp_word.w_ns + 2; j++) {
			tmp_tp[i][j] = log_fun(tmp_tp[i][j]);
		}
	}

	//use the transition matrix as the tmp_word
	tmp_word.w_tp = tmp_tp;

	return tmp_word;
}

/* 다음 state로 넘어갈때, 확률을 기록해주기 위한 함수들 */
void to_next_state(int word, int cur_s, int next_s, int time) {
	float tmp1 = total_prob[time][word][next_s];
	float tmp2 = total_prob[time - 1][word][cur_s] + logf(total_word_hmm.word_hmm_list[word].w_tp[cur_s][next_s]) + get_pes(time, word, next_s);
	
	if (tmp1 < tmp2) {
		total_prob[time][word][next_s] = tmp2;
		max_idx[time][word][next_s] = make_pair(word, cur_s);
	}
}

/* 다음 word로 넘어갈때, 확률을 기록해주기 위한 함수들 */
void to_next_word(int word, int cur_s, int next_s, int time) {
	for (int i = 0; i < 12; i++) {
		if (word == 11 && i == 11) 
			continue; 

		float tmp1 = total_prob[time][i][1];
		float tmp2 = total_prob[time - 1][word][cur_s] + logf(total_word_hmm.word_hmm_list[word].w_tp[cur_s][next_s]) + get_pes(time, i, 1) + (lambda * logf(unigram[i]));

		if (tmp1 < tmp2) {
			total_prob[time][i][1] = tmp2;
			max_idx[time][i][1] = make_pair(word, cur_s); 
		}
	}
}

/* 비터비 */
void viterbi(FILE* rst_file) {
	int input_time = testcase.size();

	// 첫번째 time : input은 반드시 "sil" 에서 시작함 - "sil"의 첫 번째 음소를 채워넣어야
	total_prob[0][11][0] = get_pes(0, 11, 0);


	// 두 번째 time ~ 마지막 time
	for (int t = 1; t < input_time; t++) {
		for (int w = 0; w < 12; w++) {
			for (int w_s = 0; w_s < total_word_hmm.word_hmm_list[w].w_ns; w_s++) {
				// 이전 state 으로부터 현재의 state로 올 수 있는 확률 = 0 일 경우 : skip
				if (total_prob[t - 1][w][w_s] == -INF)
					continue;

				/* 현재 sil 일 경우 - 단어 길이가 1인 유일한 case */
				if (w == 11) {
					if (w_s == 1) {
						to_next_state(w, w_s, w_s, t);
						to_next_state(w, w_s, w_s + 1, t);
						to_next_state(w, w_s, w_s + 2, t);
					}
					else if (w_s == 2) {
						to_next_state(w, w_s, w_s, t);
						to_next_state(w, w_s, w_s + 1, t);
					}
					else if (w_s == 3) {
						to_next_state(w, w_s, w_s, t);
						to_next_word(w, w_s, w_s + 1, t);
					}
					continue;
				}

				/* 맨 마지막 state 근처일 경우 */
				if (w_s == total_word_hmm.word_hmm_list[w].w_ns - 1) {
					// sp 바로 전 - sp를 거쳐 갈수도, sp를 거치지 않고 바로 다음 word로 넘어갈수도 있다
					to_next_state(w, w_s, w_s, t);
					to_next_state(w, w_s, w_s + 1, t);
					to_next_word(w, w_s, w_s + 1, t);

					continue;
				}
				else if (w_s == total_word_hmm.word_hmm_list[w].w_ns) {
					// sp 일 경우 
					to_next_state(w, w_s, w_s, t);
					to_next_word(w, w_s, w_s + 1, t);

					continue;
				}

				/* 그 외의 case */
				{
					to_next_state(w, w_s, w_s, t);
					to_next_state(w, w_s, w_s + 1, t);
				}
			}
		}
	}


	vector<vector<int> > rst(1000, vector<int>(2, -1));	
	float max_p = -INF;
	for (int w = 0; w < 12; w++) {
		float tmp = max_p;
		if(max_p < total_prob[input_time - 1][w][total_word_hmm.word_hmm_list[w].w_ns]){		
			rst[input_time - 1][0] = w;
			rst[input_time - 1][1] = total_word_hmm.word_hmm_list[w].w_ns;
		}
	}

	// 예외 : sil 이 가장 높은 확률일 경우
	if (max_p <total_prob[input_time - 1][11][0]) {
		rst[input_time - 1][0] = 11;
		rst[input_time - 1][1] = 0;
	}

	for (int t = input_time - 2; t >= 0; t--) {
		pair<int, int> bef_w_p = max_idx[t + 1][rst[t + 1][0]][rst[t + 1][1]];
		rst[t][0] = bef_w_p.first;
		rst[t][1] = bef_w_p.second;
	}

	
	for (int i = 0; i < input_time - 1; i++) {
		int word_num = rst[i][0];
		int state_num = rst[i][1];

		if (word_num == 11) 
			continue; 

		if (i == 0 || word_num != rst[i - 1][0] || state_num < rst[i - 1][1]) {
			switch (word_num) {
			case 0: // zero
				fprintf(rst_file, "zero\n");
				break;
			case 1: // zero
				fprintf(rst_file, "zero\n");
				break;
			case 2: // one
				fprintf(rst_file, "one\n");
				break;
			case 3: // two
				fprintf(rst_file, "two\n");
				break;
			case 4: // three
				fprintf(rst_file, "three\n");
				break;
			case 5: // four
				fprintf(rst_file, "four\n");
				break;
			case 6: // five
				fprintf(rst_file, "five\n");
				break;
			case 7: // six
				fprintf(rst_file, "six\n");
				break;
			case 8: // seven
				fprintf(rst_file, "seven\n");
				break;
			case 9: // eight
				fprintf(rst_file, "eight\n");
				break;
			case 10: // nine
				fprintf(rst_file, "nine\n");
				break;
			case 11: // oh
				fprintf(rst_file, "oh\n");
				break;
			}
		}
	}
}

/* unigram 읽어오기 */
vector<float> make_unigram() {
	vector<float> unigram(15, (float)0.0);

	FILE* fp = fopen(unigram_address, "r");

	while (!feof(fp)) {
		char buf[max_buff_len];
		fgets(buf, max_buff_len, fp);

		char* word = strtok(buf, "\n\t ");
		char* _prob = strtok(NULL, "\n\t ");
		if ((word == NULL) || (_prob == NULL))
			continue;

		unigram[word_to_int(word)] = strtof(_prob, NULL);
	}

	fclose(fp);

	return unigram;
}

/* bigram 읽어오기 */
vector<vector<float> > make_bigram() {
	vector<vector<float> > bigram(15, vector<float>(15, (float)0));

	FILE* fp = fopen(bigram_address, "r");

	while (!feof(fp)) {
		char buf[max_buff_len];
		fgets(buf, max_buff_len, fp);

		char* word1 = strtok(buf, "\n\t ");
		char* word2 = strtok(NULL, "\n\t ");
		char* _prob = strtok(NULL, "\n\t ");

		if ((word1 == NULL) || (word2 == NULL) || (_prob == NULL))
			continue;

		bigram[word_to_int(word1)][word_to_int(word2)] = strtof(_prob, NULL);
	}

	fclose(fp);
	return bigram;
}

/* dictionary 읽어오기 */
vector<vector<int> > dict_word2phone() {
	vector<vector<int> > word2phone(12);
	FILE* fp = fopen(dict_address, "r");
	printf("cc");
	while (!feof(fp)) {
		printf("dd");
		char buf[max_buff_len];
		fgets(buf, max_buff_len, fp);
		printf("aa");
		char* word = strtok(buf, "\n\t ");
		int word_idx = word_to_int(word);
		printf("bb");
		vector<int> tmp_vec;
		while (1) {
			char* phone = strtok(NULL, "\n\t ");
			if (phone == NULL)
				break;

			tmp_vec.push_back(phone_to_int(phone));
		}
		printf("cc");
		word2phone[word_idx] = tmp_vec;
	}

	fclose(fp);
	return word2phone;
}

vector<vector<float> > make_testcase() {
	FILE* fp = fopen(test_address, "r");

	int flag = 0;
	while (!feof(fp)) {
		vector<float> per_time_vec;

		char buf[max_test_len];
		fgets(buf, max_test_len, fp);

		if (flag == 0) {
			flag = 1;
			continue;
		}

		char* word = strtok(buf, "\n\t ");
		per_time_vec.push_back(strtof(word, NULL));

		while (1) {
			char* phone = strtok(NULL, "\n\t ");
			if (phone == NULL)
				break;

			per_time_vec.push_back(phone_to_int(phone));
		}

		testcase.push_back(per_time_vec);
	}
	testcase.pop_back();

	fclose(fp);
	return testcase;
}

int main() {
    // turn  txt files into usable array	
    printf("reading txt files...\n");
    word2phone = dict_word2phone();
	printf("??");
    unigram = make_unigram();
	printf("nn");
    bigram = make_bigram();
	printf("aa");
    // build hmm - word	
    printf("making word hmm...\n");
    for (int i = 0; i < 12; i++)
        total_word_hmm.word_hmm_list.push_back(make_word_hmm(i));
	printf("bb\n");
	
    // read reference files and recognize the result
    FILE* ref_file = fopen(REF_FILE, "r");
    FILE* rst_file = fopen(RST_FILE, "w");

    char buf[MAX_BUF];

    // delete #!MLF!#
    fgets(buf, MAX_BUF, ref_file);
    fputs(buf, rst_file);

    // for all lines in reference file
    while (!feof(ref_file)) {
        fgets(buf, MAX_BUF, ref_file);

        if (buf[0] != '\"')	// if read line is not file directory		
            continue;

        char test_file_name[MAX_BUF];
        fileNameRecog(buf, test_file_name);

        FILE* test_file = fopen(test_file_name, "r");
        printf("read:  %s", buf);
        if (test_file == NULL) {
            printf("file open error\n");
            printf("%s\n", test_file_name);
        }
        else {
            lab2rec(buf);
            fputs(buf, rst_file);

            // get testcase
            printf("reading test case...\n");

            int n, d;
            fscanf(test_file, "%d %d", &n, &d);
            if (d != N_DIMENSION) {
                printf("dimension error at %s\n", test_file_name);
                exit(0);
            }
            vector<vector<float> > testcase(n, vector<float>(N_DIMENSION, 0));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < N_DIMENSION; j++) {
                    fscanf(test_file, "%e", &testcase[i][j]);
                }
            }

            //viterbi
            printf("runnung viterbi...\n");
            viterbi(rst_file);
			fprintf(rst_file,".\n");

            fclose(test_file);
        }
    }
}

// int -> str
const char* int_to_str(int num) {
	if (num == 0)	return "zero";
	if (num == 1)	return "one";
	if (num == 2)	return "two";
	if (num == 3)	return "three";
	if (num == 4)	return "four";
	if (num == 5)	return "five";
	if (num == 6)	return "six";
	if (num == 7)	return "seven";
	if (num == 8)	return "eight";
	if (num == 9)	return "nine";
	if (num == 10)	return "oh";
	if (num == 11)	return "<s>";
	return "none";
}

const char* int_to_phone(int num) {
	if (num == 0)		return "f";
	if (num == 1)		return "k";
	if (num == 2)		return "n";
	if (num == 3)		return "r";
	if (num == 4)		return "s";
	if (num == 5)		return "t";
	if (num == 6)		return "v";
	if (num == 7)		return "w";
	if (num == 8)		return "z";
	if (num == 9)		return "ah";
	if (num == 10)		return "ao";
	if (num == 11)		return "ay";
	if (num == 12)		return "eh";
	if (num == 13)		return "ey";
	if (num == 14)		return "ih";
	if (num == 15)		return "iy";
	if (num == 16)		return "ow";
	if (num == 17)		return "sp";
	if (num == 18)		return "th";
	if (num == 19)		return "uw";
	if (num == 20)		return "sil";
}

// str -> int
int phone_to_int(char* s) {
	if (strcmp(s, "f") == 0)		return 0;
	if (strcmp(s, "k") == 0)		return 1;
	if (strcmp(s, "n") == 0)		return 2;
	if (strcmp(s, "r") == 0)		return 3;
	if (strcmp(s, "s") == 0)		return 4;
	if (strcmp(s, "t") == 0)		return 5;
	if (strcmp(s, "v") == 0)		return 6;
	if (strcmp(s, "w") == 0)		return 7;
	if (strcmp(s, "z") == 0)		return 8;
	if (strcmp(s, "ah") == 0)		return 9;
	if (strcmp(s, "ao") == 0)		return 10;
	if (strcmp(s, "ay") == 0)		return 11;
	if (strcmp(s, "eh") == 0)		return 12;
	if (strcmp(s, "ey") == 0)		return 13;
	if (strcmp(s, "ih") == 0)		return 14;
	if (strcmp(s, "iy") == 0)		return 15;
	if (strcmp(s, "ow") == 0)		return 16;
	if (strcmp(s, "sp") == 0)		return 17;
	if (strcmp(s, "th") == 0)		return 18;
	if (strcmp(s, "uw") == 0)		return 19;
	if (strcmp(s, "sil") == 0)		return 20;
	return -1;
}

int word_to_int(char* s) {
	if (strcmp(s, "zero") == 0)		return 0;
	if (strcmp(s, "one") == 0)		return 1;
	if (strcmp(s, "two") == 0)		return 2;
	if (strcmp(s, "three") == 0)	return 3;
	if (strcmp(s, "four") == 0)		return 4;
	if (strcmp(s, "five") == 0)		return 5;
	if (strcmp(s, "six") == 0)		return 6;
	if (strcmp(s, "seven") == 0)	return 7;
	if (strcmp(s, "eight") == 0)	return 8;
	if (strcmp(s, "nine") == 0)		return 9;
	if (strcmp(s, "oh") == 0)		return 10;
	if (strcmp(s, "<s>") == 0)		return 11;
}

int phone_to_int(const char* s) {
	if (strcmp(s, "f") == 0)		return 0;
	if (strcmp(s, "k") == 0)		return 1;
	if (strcmp(s, "n") == 0)		return 2;
	if (strcmp(s, "r") == 0)		return 3;
	if (strcmp(s, "s") == 0)		return 4;
	if (strcmp(s, "t") == 0)		return 5;
	if (strcmp(s, "v") == 0)		return 6;
	if (strcmp(s, "w") == 0)		return 7;
	if (strcmp(s, "z") == 0)		return 8;
	if (strcmp(s, "ah") == 0)		return 9;
	if (strcmp(s, "ao") == 0)		return 10;
	if (strcmp(s, "ay") == 0)		return 11;
	if (strcmp(s, "eh") == 0)		return 12;
	if (strcmp(s, "ey") == 0)		return 13;
	if (strcmp(s, "ih") == 0)		return 14;
	if (strcmp(s, "iy") == 0)		return 15;
	if (strcmp(s, "ow") == 0)		return 16;
	if (strcmp(s, "sp") == 0)		return 17;
	if (strcmp(s, "th") == 0)		return 18;
	if (strcmp(s, "uw") == 0)		return 19;
	if (strcmp(s, "sil") == 0)		return 20;
	return -1;
}

int word_to_int(const char* s) {
	if (strcmp(s, "zero") == 0)		return 0;
	if (strcmp(s, "one") == 0)		return 1;
	if (strcmp(s, "two") == 0)		return 2;
	if (strcmp(s, "three") == 0)	return 3;
	if (strcmp(s, "four") == 0)		return 4;
	if (strcmp(s, "five") == 0)		return 5;
	if (strcmp(s, "six") == 0)		return 6;
	if (strcmp(s, "seven") == 0)	return 7;
	if (strcmp(s, "eight") == 0)	return 8;
	if (strcmp(s, "nine") == 0)		return 9;
	if (strcmp(s, "oh") == 0)		return 10;
	if (strcmp(s, "<s>") == 0)		return 11;
}
