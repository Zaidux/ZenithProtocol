// zenith_core/blockchain_interface.cpp

#include "blockchain_interface.h"
#include <iostream>
#include <chrono>
#include <stdexcept>

// Mock implementation of a simple hash function.
std::string mock_sha256(const std::string& str) {
    return "hash_" + str;
}

// Mock implementation of a simple cryptographic sign function.
std::string mock_sign(const std::string& data, const std::string& key) {
    return "sig_" + mock_sha256(data + key);
}

// Mock implementation of a simple verification function.
bool mock_verify(const std::string& data, const std::string& signature, const std::string& key) {
    return signature == mock_sign(data, key);
}

// MockBlockchainLedger implementation
bool MockBlockchainLedger::add_transaction(const SignedTransaction& tx) {
    if (ledger.find(tx.data_hash) == ledger.end()) {
        ledger[tx.data_hash] = tx;
        return true;
    }
    return false;
}

SignedTransaction MockBlockchainLedger::get_transaction(const std::string& data_hash) {
    if (ledger.find(data_hash) != ledger.end()) {
        return ledger[data_hash];
    }
    return {"", "", 0};
}

// BlockchainInterface implementation
BlockchainInterface::BlockchainInterface() {
    std::cout << "Blockchain Interface initialized." << std::endl;
}

std::string BlockchainInterface::generate_hash(const std::string& data) {
    return mock_sha256(data);
}

SignedTransaction BlockchainInterface::create_signed_transaction(const std::string& data) {
    std::string data_hash = generate_hash(data);
    std::string signature = mock_sign(data_hash, mock_private_key);
    long timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    return {data_hash, signature, timestamp};
}

bool BlockchainInterface::verify_transaction(const SignedTransaction& tx, const std::string& original_data) {
    std::string data_hash = generate_hash(original_data);
    return mock_verify(data_hash, tx.signature, mock_private_key);
}

bool BlockchainInterface::add_to_blockchain(const std::string& data) {
    SignedTransaction tx = create_signed_transaction(data);
    return ledger.add_transaction(tx);
}

SignedTransaction BlockchainInterface::get_from_blockchain(const std::string& data_hash) {
    return ledger.get_transaction(data_hash);
}

PYBIND11_MODULE(blockchain_interface_cpp, m) {
    m.doc() = "C++ module for Zenith Blockchain Interface.";
    
    // Expose the SignedTransaction struct to Python.
    py::class_<SignedTransaction>(m, "SignedTransaction")
        .def(py::init<>())
        .def_readwrite("data_hash", &SignedTransaction::data_hash)
        .def_readwrite("signature", &SignedTransaction::signature)
        .def_readwrite("timestamp", &SignedTransaction::timestamp);
        
    // Expose the BlockchainInterface class.
    py::class_<BlockchainInterface>(m, "BlockchainInterface")
        .def(py::init<>())
        .def("generate_hash", &BlockchainInterface::generate_hash)
        .def("create_signed_transaction", &BlockchainInterface::create_signed_transaction)
        .def("verify_transaction", &BlockchainInterface::verify_transaction)
        .def("add_to_blockchain", &BlockchainInterface::add_to_blockchain)
        .def("get_from_blockchain", &BlockchainInterface::get_from_blockchain);
}
