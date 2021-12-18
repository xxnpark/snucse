#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include "frontend.h"
#include "backend.h"
#include "app.h"

using namespace std;

vector<string> split(string s, char c) {
    vector<string> ret;
    stringstream ss(s);
    string temp;
    while (getline(ss, temp, c)) {
        ret.push_back(temp);
    }
    return ret;
}

App::App(std::istream& is, std::ostream& os): is(is), os(os) {
    // TODO
}

void post(FrontEnd* frontend, istream& is, ostream& os) {
    frontend->post(is, os);
}

void search(FrontEnd* frontend, istream& is, ostream& os, set<string> keywords) {
    frontend->search(is, os, keywords);
}

void recommend(FrontEnd* frontend, istream& is, ostream& os, int i) {
    frontend->recommend(is, os, i);
}

bool query(std::string command, FrontEnd* frontend, istream& is, ostream& os) {
    vector<string> command_slices = split(command, ' ');
    string instruction = command_slices.front();
    if (instruction == "exit") {
        return false;
    } else if (instruction == "post") {
        post(frontend, is, os);
    } else if (instruction == "search") {
        set<string> keywords;
        for (auto iter = ++command_slices.begin(); iter != command_slices.end(); iter++) {
            keywords.insert(*iter);
        }
        search(frontend, is, os, keywords);
    } else if (instruction == "recommend") {
        recommend(frontend, is, os, stoi(command_slices.back()));
    }
    return true;
}

void App::run() {
    BackEnd* backend = new BackEnd();
    FrontEnd* frontend = new FrontEnd(backend);
    string command;
    string id_input, pw_input;
    os << "------ Authentication ------\n" << "id=";
    getline(is, id_input);
    if (!frontend->auth_id(id_input)) {
        return;
    }
    os << "passwd=";
    getline(is, pw_input);
    if (frontend->auth(id_input, pw_input)) {
        do {
            os << "-----------------------------------\n" << frontend->get_user()->id <<
            "@sns.com\n" <<
            "post : Post contents\n" <<
            "recommend <number> : recommend <number> interesting posts\n" <<
            "search <keyword> : List post entries whose contents contain <keyword>\n" <<
            "exit : Terminate this program\n" <<
            "-----------------------------------\n" <<
            "Command=";
            getline(is, command);
        } while (query(command, frontend, is, os));
    } else {
        os << "Failed Authentication.";
    }
}
