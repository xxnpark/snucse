#include "shopping_db.h"

ShoppingDB::ShoppingDB() {

}

void ShoppingDB::add_product(std::string name, int price) {
    Product* product = new Product(name, price);
    products.push_back(product);
}

bool ShoppingDB::edit_product(std::string name, int price) {
    for (Product* product : products) {
        if (product->name == name) {
            product->price = price;
            return true;
        }
    }
    return false;
}

void ShoppingDB::add_user(std::string username, std::string password, bool premium) {
    User* user;
    if (premium) {
        user = new PremiumUser(username, password);
    } else {
        user = new NormalUser(username, password);
    }
    users.push_back(user);
}

std::vector<Product*> ShoppingDB::get_products() const {
    return products;
}
std::vector<User*> ShoppingDB::get_users() const {
    return users;
}
