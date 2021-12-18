#include <vector>
#include <sstream>
#include "post.h"

Post::Post(int id, string date, string title, vector<string> contents) : id(id), date(date), title(title), contents(contents) { }

bool Post::operator<(const Post& t) const {
    return date.compare(t.date) < 0;
}
