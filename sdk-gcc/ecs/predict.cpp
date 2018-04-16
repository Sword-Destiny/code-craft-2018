#include "predict.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <ctime>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>

// #define DBG //调试

#define TRAIN_DAYS 10 //只取最近的几天的数据,13,12,11,10
#define FILL_ZERO 0 //把0记录替换为fill_zero

using namespace std;

// %Y-%m-%d %H:%M:%S
void str_to_time(const char *ph, tm *tm) {
	tm->tm_year = 0;
	tm->tm_mon = 0;
	tm->tm_mday = 0;
	tm->tm_hour = 0;
	tm->tm_min = 0;
	tm->tm_sec = 0;
	int i = 0;
	for (; i < 4; ++i) {
		tm->tm_year = tm->tm_year * 10 + ph[i] - '0';
	}
	tm->tm_year -= 1900;
	i++;
	for (; i < 7; ++i) {
		tm->tm_mon = tm->tm_mon * 10 + ph[i] - '0';
	}
	tm->tm_mon--;
	i++;
	for (; i < 10; ++i) {
		tm->tm_mday = tm->tm_mday * 10 + ph[i] - '0';
	}
	i++;
	for (; i < 13; ++i) {
		tm->tm_hour = tm->tm_hour * 10 + ph[i] - '0';
	}
	i++;
	for (; i < 16; ++i) {
		tm->tm_min = tm->tm_min * 10 + ph[i] - '0';
	}
	i++;
	for (; i < 19; ++i) {
		tm->tm_sec = tm->tm_sec * 10 + ph[i] - '0';
	}
}

//下面四行都是输入
int ph_cpu, ph_memory, ph_disk;
int flavor_type_num;
tm start_time, end_time;
time_t predict_start, predict_end;

// 优化的维度,cpu或者memory
enum resource_type {
	rt_cpu, rt_memory
};
resource_type resource;

//一条统计记录
class record {
public:
	time_t day;//日期
	int times;//某日期某flavor申请的次数
public:
	record(time_t day, int times) : day(day), times(times) {}

	~record() = default;
};

class flavor {
public:
	int fid; //flavor id
	int cpu; //flavor cpu
	int memory;//flavor memory
	vector<record> records;//统计记录
	double average;//平均数
	int predict_num;//预测得到的结果
public:
	double a, b;
public:

public:
	flavor() : fid(-1), cpu(-1), memory(-1), records(), average(-1),
			   predict_num(-1), a(0), b(0) {

	}

	~flavor() = default;

	/**
	 * 数据太少或者数据特征值为0导致无法训练,直接求平均数
	 */
	void calc_average() {
		double sum = 0.0;
		for (int day = 0; day < predict_start; ++day) {
			sum += records[day].times;
		}
		average = sum / predict_start;
	}

	/**
	 * 训练过程,数据极少时采用平均,数据较多时采用局部加权线性回归
	 */
	void train() {
		calc_average();
		fitting();
	}

	/**
	 * 线性拟合
	 */
	void fitting() {
		time_t i = 0;
		if (predict_start > TRAIN_DAYS) {
			i = predict_start - TRAIN_DAYS;
		}
		double sum_x2 = 0.0;
		double sum_y = 0.0;
		double sum_x = 0.0;
		double sum_xy = 0.0;
		for (; i < predict_start; ++i) {
			sum_x2 += records[i].day * records[i].day;
			sum_y += records[i].times;
			sum_x += records[i].day;
			sum_xy += records[i].day * records[i].times;
		}
		a = (predict_start * sum_xy - sum_x * sum_y) / (predict_start * sum_x2 - sum_x * sum_x);
		b = (sum_x2 * sum_y - sum_x * sum_xy) / (predict_start * sum_x2 - sum_x * sum_x);
	}

	// 预测
	void predict() {
		time_t p_len = predict_end - predict_start;
		double ave_sum = average * p_len;
		auto p = new double[p_len];
		double sum = 0.0;
		for (time_t i = predict_start; i < predict_end; ++i) {
			p[i - predict_start] = a * i + b;
			sum += p[i - predict_start];
		}
		correct_predict(ave_sum, sum);
#ifdef DBG
		print_predict(p);
#endif
	}

	// 模型纠偏
	void correct_predict(double ave_sum, double sum) {
		if (sum <= 0.0) {
			// 预测模型失准
			predict_num = (int) round(ave_sum);
		} else {
			if (sum > ave_sum * 10 || sum < ave_sum / 10) {
				// 预测模型失准
				predict_num = (int) round(pow(sum * ave_sum * ave_sum, 1.0 / 3));
			} else if (sum > ave_sum * 4 || sum < ave_sum / 4) {
				predict_num = (int) round(sqrt(sum * ave_sum));
			} else {
				predict_num = (int) round(sum);
			}
		}
	}

	void print_predict(const double *p) const {
		cout << "predict" << endl;
		for (int i = 0; i < predict_end - predict_start; ++i) {
			cout << p[i] << ",";
		}
		cout << endl;
		cout << "ave:" << average << endl;
		cout << "sum:" << predict_num << endl;
	}

};

flavor *flavors;

// 输出flavors,调试用的
void print_flavors() {
	for (int i = 0; i < flavor_type_num; i++) {
		cout << "flavor" << flavors[i].fid << ',' << flavors[i].cpu << ',' << flavors[i].memory << endl;
		for (auto &record : flavors[i].records) {
			cout << record.day << ',' << record.times << endl;
		}
		cout << endl;
	}
}

// 查找首个匹配的字符
int first(const char *cs, char end, char c) {
	int index = 0;
	while (cs[index] != end && cs[index] != '\n') {
		if (cs[index] == c) {
			return index;
		}
		index++;
	}
	return -1;
}

/**
 * 输入预处理
 * @param info
 */
void preprocess_info(char *info[MAX_INFO_NUM]) {
	char *ph = info[0];
	ph_cpu = 0;
	ph_memory = 0;
	ph_disk = 0;

	while (*ph != ' ') {
		ph_cpu = ph_cpu * 10 + (*ph - '0');
		ph++;
	}
	ph++;
	while (*ph != ' ') {
		ph_memory = ph_memory * 10 + (*ph - '0');
		ph++;
	}
	ph++;
	while (*ph != '\r' && *ph != '\n') {
		ph_disk = ph_disk * 10 + (*ph - '0');
		ph++;
	}

	ph = info[2];
	flavor_type_num = 0;
	while (*ph != '\r' && *ph != '\n') {
		flavor_type_num = flavor_type_num * 10 + (*ph - '0');
		ph++;
	}

	flavors = new flavor[flavor_type_num];
	for (int i = 0; i < flavor_type_num; i++) {
		ph = info[i + 3];
		int first_space = first(ph, '\r', ' ');
		ph = ph + 6;
		flavors[i].fid = 0;
		while (*ph != ' ') {
			flavors[i].fid = flavors[i].fid * 10 + (*ph - '0');
			ph++;
		}
		ph = info[i + 3] + first_space + 1;
		flavors[i].cpu = 0;
		while (*ph != ' ') {
			flavors[i].cpu = flavors[i].cpu * 10 + (*ph - '0');
			ph++;
		}
		ph++;
		flavors[i].memory = 0;
		while (*ph != '\r' && *ph != '\n') {
			flavors[i].memory = flavors[i].memory * 10 + (*ph - '0');
			ph++;
		}
		flavors[i].memory /= 1024;
	}
	ph = info[flavor_type_num + 4];
	if (ph[0] == 'c' || ph[0] == 'C') {
		resource = rt_cpu;
	} else {
		resource = rt_memory;
	}
	ph = info[flavor_type_num + 6];
	str_to_time(ph, &start_time);
	ph = info[flavor_type_num + 7];
	str_to_time(ph, &end_time);
}

void release_all() {
	delete[] flavors;
}

#define second_of_day 86400
//(24*60*60)

// 训练数据预处理
// 没有记录的,会被替换为fill_zero
void preprocess_train_data(char *data[MAX_DATA_NUM], int data_num) {
	tm flavor_time{};
	char *ph = data[0];
	int fs = first(ph, '\r', '\t');
	ph = ph + fs + 7;
	int id = 0;
	while (*ph != '\t') {
		id = id * 10 + (*ph - '0');
		ph++;
	}
	ph++;
	str_to_time(ph, &flavor_time);
	flavor_time.tm_hour = 0;
	flavor_time.tm_min = 0;
	flavor_time.tm_sec = 0;
	time_t current_second_1970 = mktime(&flavor_time);
	time_t start_day = current_second_1970 / second_of_day;
	for (int i = 0; i < data_num; ++i) {
		ph = data[i];
		fs = first(ph, '\r', '\t');
		if (fs < 1) {
			continue;//空行
		}
		ph = ph + fs + 7;
		id = 0;
		while (*ph != '\t') {
			id = id * 10 + (*ph - '0');
			ph++;
		}
		ph++;
		str_to_time(ph, &flavor_time);
		flavor_time.tm_hour = 0;
		flavor_time.tm_min = 0;
		flavor_time.tm_sec = 0;
		current_second_1970 = mktime(&flavor_time);
		for (int index = 0; index < flavor_type_num; index++) {
			if (flavors[index].fid == id) {
				time_t cday = current_second_1970 / second_of_day - start_day;
				if (flavors[index].records.empty()) {
					if (cday > 0) {
						for (time_t z = 0; z < cday; z++) {
							flavors[index].records.emplace_back(record(z, FILL_ZERO));//某种类型flavor在该天没有记录
						}
					}
					flavors[index].records.emplace_back(record(cday, 1));
				} else if (flavors[index].records[flavors[index].records.size() - 1].day == cday) {
					flavors[index].records[flavors[index].records.size() - 1].times++;
				} else {
					time_t zero_day = flavors[index].records[flavors[index].records.size() - 1].day + 1;
					if (cday > zero_day) {
						for (time_t z = zero_day; z < cday; z++) {
							flavors[index].records.emplace_back(record(z, FILL_ZERO));//某种类型flavor在该天没有记录
						}
					}
					flavors[index].records.emplace_back(record(cday, 1));
				}
				break;
			}
		}
	}
	predict_start = mktime(&start_time) / second_of_day - start_day;
	predict_end = mktime(&end_time) / second_of_day - start_day;
	for (int index = 0; index < flavor_type_num; index++) {
		if (flavors[index].records.empty()) {
			if (predict_start > 0) {
				for (time_t z = 0; z < predict_start; z++) {
					flavors[index].records.emplace_back(record(z, FILL_ZERO));//某种类型flavor在该天没有记录
				}
			}
		} else {
			time_t zero_day = flavors[index].records[flavors[index].records.size() - 1].day + 1;
			if (predict_start > zero_day) {
				for (time_t z = zero_day; z < predict_start; z++) {
					flavors[index].records.emplace_back(record(z, FILL_ZERO));//某种类型flavor在该天没有记录
				}
			}
		}
	}
#ifdef DBG
	print_flavors();
#endif
}

// 对所有数据进行训练
void train_all() {
	for (int i = 0; i < flavor_type_num; ++i) {
		flavors[i].train();
	}
}

// 对所有数据进行预测
void predict_all() {
	for (int i = 0; i < flavor_type_num; ++i) {
		flavors[i].predict();
	}
}

typedef class cell {
public:
	int value;
	int *flavor_nums;
public:
	cell() : value(0), flavor_nums(new int[flavor_type_num]) {
		memset(flavor_nums, 0, sizeof(int) * flavor_type_num);
	}

	cell(const cell &c) : value(c.value), flavor_nums(new int[flavor_type_num]) {
		memcpy(this->flavor_nums, c.flavor_nums, sizeof(int) * flavor_type_num);
	}

	cell &operator=(const cell &c) {
		if (this != &c) {
			this->value = c.value;
			memcpy(this->flavor_nums, c.flavor_nums, sizeof(int) * flavor_type_num);
		} else {
			cout << "::::" << endl;
		}
		return *this;
	}

	void init() {
		value = 0;
		memset(flavor_nums, 0, sizeof(int) * flavor_type_num);
	}

	~cell() {
		if (flavor_nums) {
			delete[] flavor_nums;
			flavor_nums = nullptr;
		}
	};
} dp_cell;


void print_cell(dp_cell *c) {
	cout << "max cpu:" << ph_cpu << "," << "max memory:" << ph_memory << endl;
	for (int i = 0; i < flavor_type_num; ++i) {
		cout << "flavor" << flavors[i].fid << ":" << c->flavor_nums[i] << "," << "cpu:" << flavors[i].cpu << ",memory:"
			 << flavors[i].memory << endl;
	}
	cout << endl;
}

dp_cell **dp;

//放置函数,动态规划
dp_cell put(int n, const int *id, const int *cpu, const int *memory, const int max_cpu, const int max_memory,
			resource_type rtype) {
	const int *value = nullptr;
	if (rtype == rt_cpu) {
		value = cpu;
	} else {
		value = memory;
	}
	// 7 56 131072
	for (int i = 0; i < n; ++i) {
		for (int j = max_cpu; j >= cpu[i]; --j) {
			for (int k = max_memory; k >= memory[i]; --k) {
				if (dp[j][k].value <= dp[j - cpu[i]][k - memory[i]].value + value[i]) {
					dp[j][k] = dp[j - cpu[i]][k - memory[i]];
					dp[j][k].value += value[i];
					dp[j][k].flavor_nums[id[i]]++;
				}
			}
		}
	}
	dp_cell res = dp[max_cpu][max_memory];
#ifdef DBG
	print_cell(&res);
#endif
	return res;
}

vector<dp_cell> final_result;

//放置函数
void put_algorithm() {
	dp = new dp_cell *[ph_cpu + 1];
	for (int i = 0; i < ph_cpu + 1; ++i) {
		dp[i] = new dp_cell[ph_memory + 1];
	}
	int n = 0;
	for (int i = 0; i < flavor_type_num; ++i) {
		n += flavors[i].predict_num;
	}
#ifdef DBG
	cout << "flavor total num:" << n << endl; //需要放置的虚拟机数量
#endif
	auto *id = new int[n];
	auto *cpu = new int[n];
	auto *memory = new int[n];
	while (n > 0) {
		int index = 0;
		for (int i = 0; i < flavor_type_num; ++i) {
			for (int j = 0; j < flavors[i].predict_num; ++j) {
				id[index] = i;
				cpu[index] = flavors[i].cpu;
				memory[index] = flavors[i].memory;
				index++;
			}
		}
		dp_cell c = put(n, id, cpu, memory, ph_cpu, ph_memory, resource);
		final_result.emplace_back(c);
		// 如果一台机器不够就再来一台
		n = 0;
		for (int i = 0; i < flavor_type_num; ++i) {
			flavors[i].predict_num -= c.flavor_nums[i];
			n += flavors[i].predict_num;
		}
		for (int i = 0; i < ph_cpu + 1; ++i) {
			for (int j = 0; j < ph_memory + 1; ++j) {
				dp[i][j].init();
			}
		}
	}
	delete[] id;
	delete[] cpu;
	delete[] memory;
	for (int i = 0; i < ph_cpu + 1; ++i) {
		delete[] dp[i];
	}
	delete[] dp;
#ifdef DBG
	cout << "ph machine num:" << final_result.size() << endl; //需要的物理机数量
#endif
}

void format_predict_data(stringstream &ss) {
	int s = 0;
	for (int i = 0; i < flavor_type_num; ++i) {
		s += flavors[i].predict_num;
	}
	ss << s << endl;
	for (int i = 0; i < flavor_type_num; ++i) {
		ss << "flavor" << flavors[i].fid << ' ' << flavors[i].predict_num << endl;
	}
	ss << endl;
}

void format_put_data(stringstream &ss) {
	ss << final_result.size();
	for (size_t i = 0; i < final_result.size(); ++i) {
		ss << endl << (i + 1);
		for (int j = 0; j < flavor_type_num; ++j) {
			if (final_result[i].flavor_nums[j] > 0) {
				ss << " flavor" << flavors[j].fid << " " << final_result[i].flavor_nums[j];
			}
		}
	}
}

void predict_server(char *info[MAX_INFO_NUM], char *data[MAX_DATA_NUM], int data_num, char *filename) {
	stringstream ss;
	preprocess_info(info);
	preprocess_train_data(data, data_num);
	train_all();
	predict_all();
	format_predict_data(ss);
	put_algorithm();
	format_put_data(ss);
#ifdef DBG
	cout << endl << ss.str().c_str() << endl;
#endif
	write_result(ss.str().c_str(), filename);
	release_all();
}