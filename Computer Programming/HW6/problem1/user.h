#ifndef PROBLEM1_USER_H
#define PROBLEM1_USER_H

#include <string>
#include <vector>
#include "product.h"

class User {
public:
    User(std::string name, std::string password);
    const std::string name;
    bool validate(std::string);
    void add_purchase_history(Product*);
    void add_to_cart(Product*);
    std::vector<Product*> get_cart() const;
    std::vector<Product*> get_purchase_history() const;
    virtual bool is_premium() {return false;};
private:
    std::string password;
    std::vector<Product*> purchase_history;
    std::vector<Product*> cart;
};

class NormalUser : public User {
public:
    NormalUser(std::string name, std::string password);
};

class PremiumUser : public User {
public:
    PremiumUser(std::string name, std::string password);
    bool is_premium() {return true;}
};

#endif //PROBLEM1_USER_H
