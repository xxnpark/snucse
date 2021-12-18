#ifndef PROBLEM2_BACKEND_H
#define PROBLEM2_BACKEND_H

#include <set>
#include <vector>
#include "config.h"
#include "user.h"
#include "post.h"

using namespace std;

struct Pair {
public:
    Post post;
    int occurence;;
    int words;
    Pair(Post, int, int);
    bool operator<(const Pair& t) const;
};

class BackEnd {
public:
    bool auth_id(string);
    User* auth(string, string);
    void post(string, string, string);
    vector<Post> recommend(string, int);
    set<Pair> search(set<string>);
    string BASE_PATH = SERVER_STORAGE_DIR;
};


#endif //PROBLEM2_BACKEND_H
