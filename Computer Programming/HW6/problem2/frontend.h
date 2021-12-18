#ifndef PROBLEM2_FRONTEND_H
#define PROBLEM2_FRONTEND_H

#include <vector>
#include <iostream>
#include "backend.h"
#include "user.h"
#include "post.h"

using namespace std;

class FrontEnd {
public:
    FrontEnd(BackEnd*);
    bool auth_id(string);
    bool auth(string, string);
    void post(std::istream& is, std::ostream& os);
    void recommend(std::istream& is, std::ostream& os, int);
    void search(std::istream& is, std::ostream& os, set<string>);
    User* get_user();
private:
    BackEnd* backend;
    User* user;
};


#endif //PROBLEM2_FRONTEND_H
