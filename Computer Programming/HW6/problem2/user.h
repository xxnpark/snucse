#ifndef PROBLEM2_USER_H
#define PROBLEM2_USER_H

#include <string>

class User {
public:
    User(std::string, std::string);
    std::string id;
private:
    std::string password;
};


#endif //PROBLEM2_USER_H
