#ifndef MARKET_DATA_H
#define MARKET_DATA_H

#include <cstdint>

struct MarketEvent
{
    uint64_t timestamp;

    float bid_price;
    float ask_price;

    uint32_t bid_quantity;
    uint32_t ask_quantity;

    float trade_price;
    uint32_t trade_quantity;

    uint8_t side;
};

#endif
