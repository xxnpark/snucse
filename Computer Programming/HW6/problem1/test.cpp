#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "shopping_db.h"
#include "admin_ui.h"
#include "client_ui.h"

#define TEST_DIRPATH "test/"

void print_OX(std::string test_name, bool is_correct) {
    std::cout << test_name << " : " << (is_correct ? "O" : "X") << std::endl;
}

bool is_space(char ch) {
    return ch == ' ' || ch == '\n' || ch == '\r';
}

void remove_space(std::string& str) {
    str.erase(std::remove_if(str.begin(), str.end(), is_space), str.end());
}

void clear_ostream_as_ostringstream(std::ostream& os) {
    std::ostringstream& oss = dynamic_cast<std::ostringstream&>(os);
    oss.str("");
    oss.clear();
}

bool compare_output(UI& ui, std::string out_filename) {
    std::ifstream ifs(TEST_DIRPATH + out_filename);
    std::ostringstream oss_answer;
    oss_answer << ifs.rdbuf();
    std::string output_answer = oss_answer.str();

    std::ostringstream& oss = dynamic_cast<std::ostringstream&>(ui.get_os());
    std::string output_app = oss.str();

    remove_space(output_answer);
    remove_space(output_app);

    return output_app == output_answer;
}

void test1(AdminUI& admin_ui) {
#if MAIN
    std::cout << std::endl << "========== Test 1 ==========" << std::endl;
#endif
#if TEST
    clear_ostream_as_ostringstream(admin_ui.get_os());
#endif

    /*
    admin_ui.add_product("tissue", 2000);
    admin_ui.add_product("chair", 20000);
    admin_ui.add_product("desk", 50000);
    admin_ui.list_products();
    admin_ui.edit_product("tissue", 3000);
    admin_ui.list_products();
    */
    admin_ui.add_product("A", 2000);
    admin_ui.add_product("B", 2000);
    admin_ui.add_product("C", 2000);
    admin_ui.add_product("D", 2000);
    admin_ui.add_product("E", 2000);

#if TEST
    bool is_correct = compare_output(admin_ui, "1.out");
    print_OX("Test 1", is_correct);
#endif
}

void test2(ClientUI& client_ui) {
#if MAIN
    std::cout << std::endl << "========== Test 2 ==========" << std::endl;
#endif
#if TEST
    clear_ostream_as_ostringstream(client_ui.get_os());
#endif

    client_ui.signup("Youngki", "hcslab", true);
    client_ui.signup("Doil", "csi", false);
    client_ui.login("Youngki", "abc");
    client_ui.login("Youngki", "hcslab");
    client_ui.add_to_cart("tissue");
    client_ui.add_to_cart("chair");
    client_ui.add_to_cart("desk");
    client_ui.list_cart_products();
    client_ui.buy_all_in_cart();
    client_ui.login("Doil", "csi");
    client_ui.logout();
    client_ui.add_to_cart("chair");
    client_ui.login("Doil", "csi");
    client_ui.buy("chair");
    client_ui.logout();
    client_ui.logout();

#if TEST
    bool is_correct = compare_output(client_ui, "2.out");
    print_OX("Test 2", is_correct);
#endif
}

void test3(ClientUI& client_ui) {
#if MAIN
    std::cout << std::endl << "========== Test 3 ==========" << std::endl;
#endif
#if TEST
    clear_ostream_as_ostringstream(client_ui.get_os());
#endif

    /*
    client_ui.signup("HyunA", "give", false);
    client_ui.signup("Kichang", "me", false);
    client_ui.signup("Hyunwoo", "jonggang", false);

    client_ui.login("HyunA", "give");
    client_ui.buy("chair");
    client_ui.buy("chair");
    client_ui.buy("chair");
    client_ui.buy("desk");
    client_ui.logout();

    client_ui.login("Hyunwoo", "jonggang");
    client_ui.buy("chair");
    client_ui.buy("tissue");
    client_ui.recommend_products();
    client_ui.logout();
    
    client_ui.login("Youngki", "hcslab");
    client_ui.recommend_products();
    client_ui.logout();
    */

    client_ui.signup("Alexa", "a", true);
    client_ui.signup("Bob", "a", false);
    client_ui.signup("Chloe", "a", false);
    client_ui.signup("David", "a", false);
    client_ui.signup("Emily", "a", false);
    client_ui.signup("Hyuna", "a", false);


    client_ui.login("Alexa", "a");
    client_ui.buy("A");
    client_ui.buy("B");
    client_ui.buy("C");
    client_ui.buy("D");
    client_ui.buy("C");
    client_ui.logout();

    client_ui.login("Bob", "a");
    client_ui.buy("A");
    client_ui.buy("C");
    client_ui.buy("A");
    client_ui.buy("E");
    client_ui.logout();

    client_ui.login("Chloe", "a");
    client_ui.buy("B");
    client_ui.buy("B");
    client_ui.buy("C");
    client_ui.buy("D");
    client_ui.buy("A");
    client_ui.buy("E");
    client_ui.logout();

    client_ui.login("David", "a");
    client_ui.add_to_cart("A");
    client_ui.add_to_cart("E");
    client_ui.add_to_cart("B");
    client_ui.buy_all_in_cart();
    client_ui.logout();

    client_ui.login("Emily", "a");
    client_ui.add_to_cart("C");
    client_ui.add_to_cart("C");
    client_ui.add_to_cart("A");
    client_ui.buy_all_in_cart();
    client_ui.logout();

    client_ui.login("Hyuna", "a");
    client_ui.buy("A");
    client_ui.buy("A");
    client_ui.buy("A");
    client_ui.buy("A");
    client_ui.buy("A");
    client_ui.buy("A");
    client_ui.logout();

    client_ui.login("Alexa", "a");
    client_ui.recommend_products();
    client_ui.logout();

    client_ui.login("Hyuna", "a");
    client_ui.recommend_products();
    client_ui.logout();

#if TEST
    bool is_correct = compare_output(client_ui, "3.out");
    print_OX("Test 3", is_correct);
#endif
}

void test4(AdminUI& admin_ui, ClientUI& client_ui) {
#if MAIN
    std::cout << std::endl << "========== Test 4 ==========" << std::endl;
#endif
    admin_ui.list_products(); //ADMIN_UI: Products: []
    admin_ui.add_product("tissue", 0); // ADMIN_UI: Invalid price.
    admin_ui.add_product("tissue", -500); // ADMIN_UI: Invalid price.
    admin_ui.add_product("tissue", 2000); // ADMIN_UI: tissue is added to the database
    // discounted price : 17996 (round up)
    admin_ui.add_product("chair", 19995); // ADMIN_UI: chair is added to the database.
    //discounted price : 44996 (round down)
    admin_ui.add_product("desk", 49996); // ADMIN_UI: desk is added to the database.
    admin_ui.list_products(); // ADMIN UI: Products: [(tissue, 2000), (chair, 19995), (desk, 49996)]
    admin_ui.edit_product("tissue", 0); // ADMIN_UI: Invalid price
    admin_ui.edit_product("tissue", 3000); // ADMIN_UI: tissue is modified from the database.
    admin_ui.edit_product("lamp", 10000); // ADMIN_UI: Invalid product name.
    admin_ui.add_product("lamp", 10000); // ADMIN_UI: lamp is added to the database.
    admin_ui.list_products(); // ADMIN UI: Products: [(tissue, 3000), (chair, 19995), (desk, 49996), (lamp, 10000)]

    client_ui.signup("premium1", "p1", true); // CLIENT_UI: premium1 is signed up.
    client_ui.signup("normal1", "n1", false); // CLIENT_UI: normal1 is signed up.
    client_ui.login("premium1", "p2"); // CLIENT_UI: Invalid username or password.
    client_ui.login("premium3", "p1"); // CLIENT_UI: Invalid username or password.
    client_ui.signup("premium2", "p2", true); // CLIENT_UI: premium2 is signed up.
    client_ui.signup("normal2", "n2", false); // CLIENT_UI: normal2 is signed up.
    client_ui.signup("premium3", "p3", true); // CLIENT_UI: premium3 is signed up.
    client_ui.signup("normal3", "n3", false); // CLIENT_UI: normal3 is signed up.
    client_ui.signup("premium4", "p4", true); // CLIENT_UI: premium4 is signed up.

    client_ui.login("premium1", "p1"); //CLIENT_UI: premium1 is logged in.
    client_ui.buy("chair"); // CLIENT_UI: Purchase completed: Price: 17996.
    client_ui.buy("desk"); // CLIENT_UI: Purchase completed: Price: 44996.
    client_ui.buy("pencil"); // CLIENT_UI: Invalid product name.
    client_ui.add_to_cart("pencil"); // CLIENT_UI: Invalid product name.
    client_ui.list_cart_products(); // CLIENT_UI: Cart: [].
    client_ui.add_to_cart("tissue"); // CLIENT_UI: tissue is added to the cart.
    client_ui.add_to_cart("chair"); // CLIENT_UI: chair is added to the cart.
    client_ui.add_to_cart("desk"); // CLIENT_UI: desk is added to the cart.
    admin_ui.edit_product("tissue", 1000); // ADMIN_UI: tissue is modified from the database.
    client_ui.list_cart_products(); // CLIENT_UI: Cart: [(tissue, 900), (chair, 17996), (desk, 44996)].
    admin_ui.edit_product("tissue", 500); // ADMIN_UI: tissue is modified from the database.
    client_ui.buy_all_in_cart(); // CLIENT_UI: Cart purchase completed. Total price: 63442.
    client_ui.buy_all_in_cart(); // CLIENT_UI: Cart purchase completed. Total price: 0.
    client_ui.logout(); // CLIENT_UI: premium1 is logged out.

    client_ui.login("normal1", "n1"); //CIENT_UI: normal1 is logged in.
    client_ui.add_to_cart("tissue"); // CLIENT_UI: tissue is added to the cart.
    client_ui.add_to_cart("chair"); // CLIENT_UI: chair is added to the cart.
    client_ui.add_to_cart("chair"); // CLIENT_UI: chair is added to the cart.
    client_ui.add_to_cart("desk"); // CLIENT_UI: desk is added to the cart.
    client_ui.add_to_cart("desk"); // CLIENT_UI: desk is added to the cart.
    client_ui.buy_all_in_cart(); // CLIENT_UI: Cart purchase completed. Total price: 140482.
    client_ui.logout(); // CLIENT_UI: normal1 is logged out.

    client_ui.login("premium2", "p2"); //CLIENT_UI: premium2 is logged in.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 450.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 450.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 450.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 450.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 450.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 450.
    client_ui.buy("lamp"); // CLIENT_UI: Purchase completed. Price: 9000.
    client_ui.logout(); // CLIENT_UI: premium2 is logged out.

    client_ui.login("normal2", "n2"); //CLIENT_UI: normal2 is logged in.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 500.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 500.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 500.
    client_ui.buy("lamp"); // CLIENT_UI: Purchase completed. Price: 10000.
    client_ui.logout(); // CLIENT_UI: normal2 is logged out.

    client_ui.login("premium3", "p3"); //CLIENT_UI: premium3 is logged in.
    client_ui.buy("desk"); // CLIENT_UI: Purchase completed. Price: 44996.
    client_ui.buy("chair"); // CLIENT_UI: Purchase completed. Price: 17996.
    client_ui.buy("tissue"); // CLIENT_UI: Purchase completed. Price: 450.
    client_ui.logout(); // CLIENT_UI: premium 3 is logged out.

    client_ui.login("normal1", "n1"); //CLIENT_UI: normal1 is logged in.
    admin_ui.edit_product("tissue", 1000); // ADMIN_UI: tissue is modified from the database.
    client_ui.recommend_products(); //CLIENT_UI: Recommended products : [(desk, 49996), (chair, 19995), (tissue, 1000)]
    client_ui.logout(); // CLIENT_UI: normal1 is logged out.

    client_ui.login("normal2", "n2"); //CLIENT_UI: normal2 is logged in.
    client_ui.recommend_products(); // CLIENT_UI: Recommended products: [(lamp, 10000), (tissue, 1000)]
    client_ui.logout(); // CLIENT_UI: normal2 is logged out.

    client_ui.login("normal3", "n3"); //CLIENT_UI: normal3 is logged in.
    client_ui.recommend_products(); // CLIENT_UI: Recommended products: []
    client_ui.logout(); //CLIENT_UI: normal3 is logged out.

    client_ui.login("premium1", "p1"); //CLIENT_UI: premium1 is logged in.
    client_ui.recommend_products(); // CLIENT_UI: Recommended products: [(lamp, 9000), (desk, 44996), (tissue, 900)]
    client_ui.logout(); //CLIENT_UI: premium1 is logged out.

    client_ui.login("premium2", "p2"); //CLIENT_UI: premium2 is logged in.
    admin_ui.edit_product("tissue", 10000); // ADMIN_UI: tissue is modified from the database.
    client_ui.recommend_products(); // CLIENT_UI: Recommended products: [(lamp, 9000), (desk, 44996), (tissue, 9000)]
    client_ui.logout(); //CLIENT_UI: premium2 is logged out.
}

int main() {
#if MAIN
    std::ostream& os = std::cout;
#endif
#if TEST
    std::ostringstream os;
#endif
    ShoppingDB db;
    AdminUI admin_ui(db, os);
    ClientUI client_ui(db, os);
//    test1(admin_ui);
//    test2(client_ui);
//    test3(client_ui);
    test4(admin_ui, client_ui);
}
