#include <iostream>
#include <vector>
#include <random>
#include "../include/market_data.h"

std::vector<MarketEvent> generateMarketData(int N)
{
    std::vector<MarketEvent> events(N);

    std::mt19937 rng(42);

    std::uniform_real_distribution<float> price(99.0, 101.0);
    std::uniform_int_distribution<uint32_t> quantity(100, 1000);
    std::uniform_int_distribution<int> side(0, 1);

    for (int i = 0; i < N; i++)
    {
        float mid = price(rng);
        float spread = 0.02f;

        events[i].timestamp = i * 1000;

        events[i].bid_price = mid - spread / 2;
        events[i].ask_price = mid + spread / 2;

        events[i].bid_quantity = quantity(rng);
        events[i].ask_quantity = quantity(rng);

        events[i].trade_price = mid;
        events[i].trade_quantity = quantity(rng);

        events[i].side = side(rng);
    }

    return events;
}

int main()
{
    int N = 10;

    auto events = generateMarketData(N);

    std::cout << "Synthetic HFT Market Data\n\n";

    for (int i = 0; i < N; i++)
    {
        std::cout
            << "Event " << i
            << " | Bid: " << events[i].bid_price
            << " | Ask: " << events[i].ask_price
            << " | Bid Qty: " << events[i].bid_quantity
            << " | Ask Qty: " << events[i].ask_quantity
            << "\n";
    }

    return 0;
}
