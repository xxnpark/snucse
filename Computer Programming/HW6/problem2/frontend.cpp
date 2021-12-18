#include <set>
#include "frontend.h"

FrontEnd::FrontEnd(BackEnd* backend) : backend(backend) { }

bool FrontEnd::auth_id(string id_input) {
    return backend->auth_id(id_input);
}

bool FrontEnd::auth(string id_input, string pw_input) {
    User* user = backend->auth(id_input, pw_input);
    if (user == nullptr) {
        return false;
    } else {
        this->user = user;
        return true;
    }
}

void FrontEnd::post(istream& is, ostream& os) {
    string title, content, contents;

    os << "-----------------------------------\nNew Post\n* Title=";
    getline(is, title);

    os << "* Content\n>";
    getline(is, content);
    contents += "\n" + content;
    os << ">";
    while (true) {
        getline(is, content);
        if (content.empty()) {
            break;
        }
        contents += "\n" + content;
        os << ">";
    }

    backend->post(user->id, title, contents);
}

void FrontEnd::recommend(std::istream&is, std::ostream& os, int i) {
    vector<Post> posts = backend->recommend(user->id, i);
    for (Post post : posts) {
        os << "-----------------------------------\n" <<
              "id: " << post.id << "\n" <<
              "created at: " << post.date << "\n" <<
              "title: " << post.title << "\n" <<
              "content:\n";
        for (string content : post.contents) {
            os << content << "\n";
        }
    }
}

void FrontEnd::search(std::istream& is, std::ostream& os, set<string> keywords) {
    set<Pair> posts = backend->search(keywords);
    if (!posts.empty()) {
        os << "-----------------------------------\n";
    }

    auto iter = posts.begin();
    for (int count = 0; count < 10; count++) {
        if (iter == posts.end()) {
            break;
        }
        os << "id: " << (*iter).post.id <<
              ", created at: " << (*iter).post.date <<
              ", title: " << (*iter).post.title << "\n";
        iter++;
    }
}

User* FrontEnd::get_user() {
    return user;
}