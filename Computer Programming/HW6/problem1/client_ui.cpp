#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include "client_ui.h"
#include "product.h"
#include "user.h"
#include "user_similarity.h"

ClientUI::ClientUI(ShoppingDB &db, std::ostream& os) : UI(db, os), current_user() { }

void ClientUI::signup(std::string username, std::string password, bool premium) {
    db.add_user(username, password, premium);
    os << "CLIENT_UI: " << username << " is signed up.\n";
}

void ClientUI::login(std::string username, std::string password) {
    if (current_user != nullptr) {
        os << "CLIENT_UI: Please logout first.\n";
        return;
    }

    for (User* user : db.get_users()) {
        if (user->name == username && user->validate(password)) {
            os << "CLIENT_UI: " << username << " is logged in.\n";
            current_user = user;
            return;
        }
    }

    os << "CLIENT_UI: Invalid username or password.\n";
}

void ClientUI::logout() {
    if (current_user == nullptr) {
        os << "CLIENT_UI: There is no logged-in user.\n";
    } else {
        os << "CLIENT_UI: " << current_user->name << " is logged out.\n";
        current_user = nullptr;
    }
}

void ClientUI::add_to_cart(std::string product_name) {
    if (current_user == nullptr) {
        os << "CLIENT_UI: Please login first.\n";
        return;
    }

    for (Product* product : db.get_products()) {
        if (product->name == product_name) {
            current_user->add_to_cart(product);
            os << "CLIENT_UI: " << product_name << " is added to the cart.\n";
            return;
        }
    }
    os << "CLIENT_UI: Invalid product name.\n";
}

void ClientUI::list_cart_products() {
    if (current_user == nullptr) {
        os << "CLIENT_UI: Please login first.\n";
        return;
    }

    std::vector<Product*> products = current_user->get_cart();

    os << "CLIENT_UI: Cart: [";
    for (Product* product : products) {
        if (product != *products.begin()) {
            os << ", ";
        }
        if (current_user->is_premium()) {
            os << "(" << product->name << ", " << round(product->price * 0.9) << ")";
        } else {
            os << "(" << product->name << ", " << product->price << ")";
        }
    }
    os << "]\n";
}

void ClientUI::buy_all_in_cart() {
    if (current_user == nullptr) {
        os << "CLIENT_UI: Please login first.\n";
        return;
    }

    std::vector<Product*> products = current_user->get_cart();
    int total_price = 0;

    if (!products.empty()) {
        for (auto iter = products.begin(); iter != products.end();) {
            current_user->add_purchase_history(*iter);
            if (current_user->is_premium()) {
                total_price += round((*iter)->price * 0.9);
            } else {
                total_price += (*iter)->price;
            }
            iter = products.erase(iter);
        }
    }
    os << "CLIENT_UI: Cart purchase completed. Total price: " << total_price << ".\n";
}

void ClientUI::buy(std::string product_name) {
    if (current_user == nullptr) {
        os << "CLIENT_UI: Please login first.\n";
        return;
    }

    for (Product* product : db.get_products()) {
        if (product->name == product_name) {
            current_user->add_purchase_history(product);
            if (current_user->is_premium()) {
                os << "CLIENT_UI: Purchase completed. Price: " << round(product->price * 0.9) << ".\n";
            } else {
                os << "CLIENT_UI: Purchase completed. Price: " << product->price << ".\n";
            }
            return;
        }
    }
    os << "CLIENT_UI: Invalid product name.\n";
}

void ClientUI::recommend_products() {
    if (current_user == nullptr) {
        os << "CLIENT_UI: Please login first.\n";
        return;
    }

    std::vector<Product*> purchase_history = current_user->get_purchase_history();

    os << "CLIENT_UI: Recommended products: [";
    if (current_user->is_premium()) {
        std::vector<User*> users = db.get_users();
        std::set<UserSimilarity> other_users;

        if (!users.empty()) {
            for (auto iter = users.begin(); iter != users.end(); ++iter) {
                if (*iter == current_user) {
                    continue;
                }
                int similarity = 0;
                for (Product* product : (*iter)->get_purchase_history()) {
                    if (std::find(purchase_history.begin(),purchase_history.end(), product) != purchase_history.end()) {
                        similarity++;
                    }
                }
                UserSimilarity user_similarity(*iter, iter - users.begin(), similarity);
                other_users.insert(user_similarity);
            }
        }

        std::vector<Product*> recommends;
        auto iter = other_users.begin();
        while (recommends.size() < 3 && iter != other_users.end()) {
            if ((*iter).user->get_purchase_history().size() == 0) {
                iter++;
                continue;
            }
            Product* latest_purchase = *((*iter).user->get_purchase_history().end() - 1);
            if (std::find(recommends.begin(), recommends.end(), latest_purchase) == recommends.end()) {
                recommends.push_back(latest_purchase);
                if (iter != other_users.begin()) {
                    os << ", ";
                }
                os << "(" << latest_purchase->name << ", " << round(latest_purchase->price * 0.9) << ")";
            }
            iter++;
        }
    } else {
        if (!purchase_history.empty()) {
            std::vector<Product*> recommends;
            auto iter = purchase_history.end() - 1;
            while (recommends.size() < 3 && iter >= purchase_history.begin()) {
                if (std::find(recommends.begin(), recommends.end(), *iter) == recommends.end()) {
                    recommends.push_back(*iter);
                    if (iter < purchase_history.end() - 1) {
                        os << ", ";
                    }
                    os << "(" << (*iter)->name << ", " << (*iter)->price << ")";
                }
                iter--;
            }
        }
    }
    os << "]\n";
}
