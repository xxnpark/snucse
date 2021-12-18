#ifndef PROBLEM1_USER_SIMILARITY_H
#define PROBLEM1_USER_SIMILARITY_H

#include "user.h"

class UserSimilarity {
public:
    UserSimilarity(User* user, int registered, int similarity);
    User* user;
    int registered;
    int similiarity;
    bool operator<(const UserSimilarity& t) const;
};

#endif //PROBLEM1_USER_SIMILARITY_H
