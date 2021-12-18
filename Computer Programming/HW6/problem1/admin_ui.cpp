#include <vector>
#include "admin_ui.h"

AdminUI::AdminUI(ShoppingDB &db, std::ostream& os): UI(db, os) { }

void AdminUI::add_product(std::string name, int price) {
    if (price > 0) {
        os << "ADMIN_UI: " << name << " is added to the database.\n";
        db.add_product(name, price);
    } else {
        os << "ADMIN_UI: Invalid price.\n";
    }
}

void AdminUI::edit_product(std::string name, int price) {
    if (price > 0) {
        if (db.edit_product(name, price)) {
            os << "ADMIN_UI: " << name << " is modified from the database.\n";
        } else {
            os << "ADMIN_UI: Invalid product name.\n";
        }
    } else {
        os << "ADMIN_UI: Invalid price.\n";
    }
}

void AdminUI::list_products() {
    std::vector<Product*> products = db.get_products();
    os << "ADMIN_UI: Products: [";
    for (Product* product : products) {
        if (product != *products.begin()) {
            os << ", ";
        }
        os << "(" << product->name << ", " << product->price << ")";
    }
    os << "]\n";
}
