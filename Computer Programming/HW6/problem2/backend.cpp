#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include "backend.h"

vector<string> splith(string s, char c) {
    vector<string> ret;
    stringstream ss(s);
    string temp;
    while (getline(ss, temp, c)) {
        ret.push_back(temp);
    }
    return ret;
}

Pair::Pair(Post post, int occurence, int words) : post(post), occurence(occurence), words(words) {}
bool Pair::operator<(const Pair& t) const {
    if (occurence == t.occurence) return words > t.words;
    return occurence > t.occurence;
}

bool BackEnd::auth_id(string id_input) {
    for (const auto & iter : filesystem::directory_iterator(BASE_PATH)) {
        if (iter.path().string() == BASE_PATH + id_input) {
            return true;
        }
    }
    return false;
}
User* BackEnd::auth(string id_input, string pw_input) {
    string str;
    ifstream user_pw_file(BASE_PATH + id_input + "/password.txt");
    if (getline(user_pw_file, str)) {
        transform(str.begin(), str.end(), str.begin(), ::tolower);
        transform(pw_input.begin(), pw_input.end(), pw_input.begin(), ::tolower);
        if (str.compare(pw_input) == 0) {
            return new User(id_input, pw_input);
        }
    }
    return nullptr;
}

void BackEnd::post(string id, string title, string contents) {
    set<int> post_ids;
    if (!filesystem::is_empty(BASE_PATH)) {
        for (const auto &iter1: filesystem::directory_iterator(BASE_PATH)) {
            if (iter1.path().string() == BASE_PATH + ".DS_Store") continue;
            if (filesystem::is_empty(iter1.path().string() + "/post/")) continue;
            for (const auto &iter2: filesystem::directory_iterator(iter1.path().string() + "/post/")) {
                post_ids.insert(stoi(iter2.path().string().substr(iter1.path().string().size() + 6,
                                                                  iter2.path().string().size() -
                                                                  iter1.path().string().size() - 10)));
            }
        }
    }

    int post_id = post_ids.empty() ? 0 : *(--post_ids.end()) + 1;

    ofstream user_post_file(BASE_PATH + id + "/post/" + to_string(post_id) + ".txt");

    auto now = chrono::system_clock::now();
    time_t time = chrono::system_clock::to_time_t(now);
    user_post_file << std::put_time(std::localtime(&time), "%Y/%m/%d %X") << "\n" << title << "\n" << contents;
}

vector<Post> BackEnd::recommend(string id, int i) {
    set<Post> posts;
    vector<Post> ret;
    string str;
    ifstream user_pw_file(BASE_PATH + id + "/friend.txt");
    while (getline(user_pw_file, str)) {
        for (const auto &iter: filesystem::directory_iterator(BASE_PATH + str + "/post/")) {
            int id = stoi(iter.path().string().substr((BASE_PATH + str).size() + 6,
                                                   iter.path().string().size() - (BASE_PATH + str).size() - 10));
            string date, title, temp;
            vector<string> contents;
            ifstream post_file(iter.path().string());
            getline(post_file, date);
            getline(post_file, title);
            getline(post_file, temp);
            while (getline(post_file, temp)) {
                contents.push_back(temp);
            }
            posts.insert(Post(id, date, title, contents));
        }
    }

    auto iter = --posts.end();
    for (int count = 0; count < i; count++) {
        ret.push_back(*iter);
        if (iter-- == posts.begin()) {
            break;
        }
    }

    return ret;
}

set<Pair> BackEnd::search(set<string> keywords) {
    set<Pair> posts;
    for (const auto &iter1: filesystem::directory_iterator(BASE_PATH)) {
        if (iter1.path().string() == BASE_PATH + ".DS_Store") continue;
        for (const auto &iter2: filesystem::directory_iterator(iter1.path().string() + "/post/")) {
            int id = stoi(iter2.path().string().substr(iter1.path().string().size() + 6,
                                              iter2.path().string().size() - iter1.path().string().size() - 10));
            string date, title, temp;
            vector<string> contents;
            ifstream post_file(iter2.path().string());
            getline(post_file, date);
            getline(post_file, title);
            getline(post_file, temp);
            while (getline(post_file, temp)) {
                contents.push_back(temp);
            }

            int occurence = 0;
            int words = 0;
            for (string word : splith(title, ' ')) {
                for (string keyword : keywords) {
                    if (word == keyword) {
                        occurence++;
                    }
                }
            }
            for (string content : contents) {
                for (string word : splith(content, ' ')) {
                    words++;
                    for (string keyword : keywords) {
                        if (word == keyword) {
                            occurence++;
                        }
                    }
                }
            }
            if (occurence > 0) {
                posts.insert(Pair(Post(id, date, title, contents), occurence, words));
            }
        }
    }

    return posts;
}
