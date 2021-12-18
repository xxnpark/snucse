#include "user_similarity.h"

UserSimilarity::UserSimilarity(User *user, int registered, int similarity) : user(user), registered(registered),similiarity(similarity) {}

bool UserSimilarity::operator<(const UserSimilarity &t) const {
    if (similiarity == t.similiarity) return registered < t.registered;
    return similiarity > t.similiarity;
}
