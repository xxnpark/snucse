#ifndef PROBLEM2_POST_H
#define PROBLEM2_POST_H

#include <string>

using namespace std;

class Post {
public:
    Post(int, string, string, vector<string>);
    bool operator<(const Post&) const;
    int id;
    string date;
    string title;
    vector<string> contents;

};


#endif //PROBLEM2_POST_H
