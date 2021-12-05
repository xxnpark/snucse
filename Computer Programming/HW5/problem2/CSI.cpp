#include "CSI.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <climits>

using namespace std;

Complex::Complex(): real(0), imag(0) {}

CSI::CSI(): data(nullptr), num_packets(0), num_channel(0), num_subcarrier(0) {}

CSI::~CSI() {
    if(data) {
        for(int i = 0 ; i < num_packets; i++) {
            delete[] data[i];
        }
        delete[] data;
    }
}

int CSI::packet_length() const {
    return num_channel * num_subcarrier;
}

void CSI::print(std::ostream& os) const {
    for (int i = 0; i < num_packets; i++) {
        for (int j = 0; j < packet_length(); j++) {
            os << data[i][j] << ' ';
        }
        os << std::endl;
    }
}

std::ostream& operator<<(std::ostream &os, const Complex &c) {
    if (c.imag >= 0) {
        return os << c.real << "+" << c.imag << "i";
    } else {
        return os << c.real << c.imag << "i";
    }
}

void read_csi(const char* filename, CSI* csi) {
    ifstream file(filename);
    string packets, channels, subcarriers;
    getline(file, packets); getline(file, channels); getline(file, subcarriers);
    int num_packets = stoi(packets), num_channel = stoi(channels), num_subcarrier = stoi(subcarriers);

    csi->data = new Complex*[num_packets];

    for (int i = 0; i < num_packets; i++) {
        csi->data[i] = new Complex[num_channel*num_subcarrier];
        for (int j = 0; j < num_subcarrier; j++) {
            for (int k = 0; k < num_channel; k++) {
                Complex c;
                string real, imag;
                getline(file, real); getline(file, imag);
                c.real = stoi(real); c.imag = stoi(imag);
                csi->data[i][j+k*num_subcarrier] = c;
            }
        }
    }

    csi->num_packets = num_packets; csi->num_channel = num_channel; csi->num_subcarrier = num_subcarrier;

    file.close();
}

double** decode_csi(CSI* csi) {
    int num_packets = csi->num_packets, num_channel = csi->num_channel, num_subcarrier = csi->num_subcarrier;

    double** decoded_csi = new double*[num_packets];

    for (int i = 0; i < num_packets; i++) {
        decoded_csi[i] = new double[num_channel*num_subcarrier];
        for (int j = 0; j < num_subcarrier; j++) {
            for (int k = 0; k < num_channel; k++) {
                int real = csi->data[i][j+k*num_subcarrier].real, imag = csi->data[i][j+k*num_subcarrier].imag;
                decoded_csi[i][j+k*num_subcarrier] = sqrt(real * real + imag * imag);
            }
        }
    }

    return decoded_csi;
}

double* get_med(double** decoded_csi, int num_packets, int packet_length) {
    double* medians = new double[num_packets];
    int index = packet_length / 2;

    for (int i = 0; i < num_packets; i++) {
        double* packet = new double[packet_length];
        for (int j = 0; j < packet_length; j++) {
            packet[j] = decoded_csi[i][j];
        }
        sort(packet, packet + packet_length);

        if (packet_length % 2 == 1) {
            medians[i] = packet[index];
        } else {
            medians[i] = (packet[index-1] + packet[index]) / 2;
        }
    }

    return medians;
}

double breathing_interval(double** decoded_csi, int num_packets) {
    double eps = pow(10, -10);

    int peaks[num_packets];
    int peak_count = 0;

    for (int i = 0; i < num_packets; i++) {
        double curr = decoded_csi[i][0];
        double prev1 = INT_MIN, prev2 = INT_MIN, foll1 = INT_MIN, foll2 = INT_MIN;

        if (i > 0) {
            prev1 = decoded_csi[i-1][0];
            if (i > 1) {
                prev2 = decoded_csi[i-2][0];
            }
        }
        if (i < num_packets - 1) {
            foll1 = decoded_csi[i+1][0];
            if (i < num_packets - 2) {
                foll2 = decoded_csi[i+2][0];
            }
        }

        if (curr > prev1 + eps && curr > prev2 + eps && curr > foll1 + eps && curr > foll2 + eps) {
            peaks[peak_count++] = i;
        }
    }

    if (peak_count <= 1) {
        return num_packets;
    }

    double interval_sum = 0.0;

    for (int i = 1; i < peak_count; i++) {
        interval_sum += peaks[i] - peaks[i-1];
    }

    return interval_sum / (peak_count - 1);
}
