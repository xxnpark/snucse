#include <iostream>
#include <ctime>
#include "player.h"

void round(Player &a, Player &b);

int main() {
    std::srand(std::time(nullptr));

    Player a, b;
    a.add_monster(fireMon);
    a.add_monster(fireMon);
    a.add_monster(waterMon);
    b.add_monster(waterMon);
    b.add_monster(grassMon);
    b.add_monster(grassMon);

    std::cout << "Game start! Player A: " << a.get_total_hp() << ", Player B: " << b.get_total_hp() << std::endl;

    for (int i = 1;; i++) {
        round(a, b);
        std::cout << "Round " << i << ": " << a.get_total_hp() << " " << b.get_total_hp() << std::endl;
        if (b.get_num_monsters() == 0) {
            std::cout << "Player a won the game!" << std::endl;
            break;
        } else if (a.get_num_monsters() == 0) {
            std::cout << "Player b won the game!" << std::endl;
            break;
        }
    }
    return 0;
}

void round(Player &a, Player &b) {
    Monster *a_mon = a.select_monster();
    Monster *b_mon = b.select_monster();

    if (a_mon->get_speed() > b_mon->get_speed()) {
        a_mon->attack(b_mon);
        if (b_mon->get_hp() <= 0) {
            std::cout << "Player B's " << *b_mon << " fainted!" << std::endl;
            b.delete_monster(b_mon);
        } else {
            b_mon->attack(a_mon);
            if (a_mon->get_hp() <= 0) {
                std::cout << "Player A's " << *a_mon << " fainted!" << std::endl;
                a.delete_monster(a_mon);
            }
        }
    } else {
        b_mon->attack(a_mon);
        if (a_mon->get_hp() <= 0) {
            std::cout << "Player A's " << *a_mon << " fainted!" << std::endl;
            a.delete_monster(a_mon);
        } else {
            a_mon->attack(b_mon);
            if (a_mon->get_hp() <= 0) {
                std::cout << "Player B's " << *b_mon << " fainted!" << std::endl;
                b.delete_monster(b_mon);
            }
        }
    }
}
