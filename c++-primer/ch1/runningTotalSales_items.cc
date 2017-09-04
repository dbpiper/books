#include <iostream>
#include "Sales_item.h"

int main()
{
    Sales_item sum, value;
    while (std::cin >> value)
        sum += value;
    std::cout << sum << std::endl; // print their sum
    return 0;
}
