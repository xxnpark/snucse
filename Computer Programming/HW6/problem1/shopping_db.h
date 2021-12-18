#ifndef PROBLEM1_SHOPPING_DB_H
#define PROBLEM1_SHOPPING_DB_H

#include <string>
#include <vector>
#include "user.h"
#include "product.h"

class ShoppingDB {
public:
    ShoppingDB();
    void add_product(std::string, int);
    bool edit_product(std::string, int);
    void add_user(std::string, std::string, bool);
    std::vector<Product*> get_products() const;
    std::vector<User*> get_users() const;
private:
    std::vector<User*> users;
    std::vector<Product*> products;
};

#endif //PROBLEM1_SHOPPING_DB_H
