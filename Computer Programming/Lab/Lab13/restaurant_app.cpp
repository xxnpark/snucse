#include <algorithm>
#include "restaurant_app.h"

#include <iostream>
#include <iomanip>


void RestaurantApp::rate(string target, int rate) {
    shared_ptr<vector<int>> ratings = find_restaurant(target);

    if (!ratings) {
        vector<int> ratings;
        ratings.push_back(rate);
        restaurants.insert(std::pair<string, shared_ptr<vector<int>>> (target, std::make_shared<vector<int>>(ratings)));
        return;
    }

    ratings->push_back(rate);
    std::sort(ratings->begin(), ratings->end());
}


void RestaurantApp::list() {
    for (auto item : restaurants) {
        std::cout << item.first << " ";
    }
    std::cout << std::endl;
}


void RestaurantApp::show(string target) {
    shared_ptr<vector<int>> ratings = find_restaurant(target);

    if (!ratings) {
        std::cout << target << " does not exist." << std::endl;
        return;
    }

    for (int rate : *ratings) {
        std::cout << rate << " ";
    }
    std::cout << std::endl;
}


void RestaurantApp::ave(string target) {
    shared_ptr<vector<int>> ratings = find_restaurant(target);

    if (!ratings) {
        std::cout << target << " does not exist." << std::endl;
        return;
    }

    double average = 0;
    for (int rate: *ratings) {
        average += rate;
    }

    average /= double(ratings->size());

    std::cout << std::fixed << std::setprecision(2) << average << std::endl;
}

void RestaurantApp::del(string target, int rate) {
    shared_ptr<vector<int>> ratings = find_restaurant(target);

    if (!ratings) {
        std::cout << target << " does not exist." << std::endl;
        return;
    }

    ratings->erase(std::remove(ratings->begin(), ratings->end(), rate), ratings->end());
}


void RestaurantApp::cheat(string target, int rate) {
    shared_ptr<vector<int>> ratings = find_restaurant(target);

    if (!ratings) {
        std::cout << target << " does not exist." << std::endl;
        return;
    }

    ratings->erase(std::remove_if(ratings->begin(), ratings->end(), [rate](int val){return val < rate;}), ratings->end());
}


shared_ptr<vector<int>> RestaurantApp::find_restaurant(string target) {
    auto it = restaurants.find(target);
    if (it == restaurants.end()) {
        return nullptr;
    }
    return restaurants[target];
}
