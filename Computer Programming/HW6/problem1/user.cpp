#include "user.h"

User::User(std::string name, std::string password): name(name), password(password) {

}

bool User::validate(std::string password_input) {
    if (password == password_input) {
        return true;
    } else {
        return false;
    }
}

void User::add_purchase_history(Product* product) {
    purchase_history.push_back(product);
}

void User::add_to_cart(Product* product) {
    cart.push_back(product);
}

std::vector<Product*> User::get_cart() const {
    return cart;
}

std::vector<Product*> User::get_purchase_history() const {
    return purchase_history;
}

NormalUser::NormalUser(std::string name, std::string password): User(name, password) {

}

PremiumUser::PremiumUser(std::string name, std::string password): User(name, password) {

}
