# PyNetQA - IP Network QA Tool

A professional network quality assurance testing tool that performs comprehensive measurements of network performance metrics including bandwidth, jitter, MTU, and connectivity. Generate detailed HTML reports with visualizations and system information for network validation and documentation.

## Features

### Network Tests
- **Performance Testing**: Measures network bandwidth using iPerf3
- **Jitter Analysis**: Measures network jitter and stability
- **MTU Testing**: Determines maximum supported MTU sizes
- **Connectivity Testing**: Checks basic network connectivity and packet loss

### Reporting
- **HTML Reports**: Clean, professional reports with visualizations
- **JSON Data**: Raw test data in JSON format
- **Visualizations**: 
  - Performance graphs over time
  - Jitter analysis plots with statistics
  - MTU test results with success/failure indicators
- **System Information**: 
  - OS details and version
  - Network configuration
  - Test environment parameters
  - Client/Server information

### Quality Assurance
- Unique test IDs (UUID) for traceability
- Timestamp verification
- Complete test environment logging
- Test result verification and validation

## Prerequisites

- Python 3.6 or higher
- iPerf3 installed on both client and server machines
- Network connectivity between test machines

### Required Python Packages
```text
matplotlib>=3.4.0
seaborn>=0.11.0
jinja2>=3.0.0
numpy>=1.20.0
pandas>=1.3.0
```

## Installation

### Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/wolfgang-azevedo/pynetqa.git
cd pynetqa
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Offline Installation

For systems without internet access, follow these steps:

1. On a system with internet access:
```bash
# Create a directory for offline installation files
mkdir pynetqa_offline
cd pynetqa_offline

# Download Python packages
pip download -r requirements.txt -d python_packages/

# Create the offline package structure
mkdir -p offline_package/
cp -r python_packages offline_package/
cp network_test.py offline_package/
cp requirements.txt offline_package/
cp -r img offline_package/
```

2. On the target system:
```bash
# Extract the offline package
cd pynetqa_offline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install packages from offline directory
pip install --no-index --find-links python_packages -r requirements.txt
```

## Usage

### Starting the iPerf3 Server
On the server machine:
```bash
iperf3 -s
```

### Running Tests
On the client machine:
```bash
python network_test.py --server <server_ip> --iperf-path <path_to_iperf3>

# Example:
python network_test.py --server 172.25.56.40 --iperf-path ./bin/iperf3.exe
```

### Command Line Options
```
--server, -s    : Server IP address (required)
--iperf-path    : Path to iPerf3 executable (optional)
--log-dir       : Directory for log files (optional)
--test          : Specific test to run [all|connectivity|performance|jitter|mtu] (optional)
```

## Output Structure

PyNetQA generates several output files in the `network_tests` directory:

```
network_tests/
├── report_YYYYMMDD_HHMMSS/
│   ├── img/
│   │   └── logo.png
│   ├── performance.png
│   ├── jitter.png
│   ├── mtu.png
│   ├── report.html
│   └── results.json
└── network_test_YYYYMMDD_HHMMSS.log
```

### Report Contents
- Executive Summary
- Detailed Test Results
- System Information
- Performance Graphs
- Jitter Analysis
- MTU Test Results
- Test Configuration
- Unique Test Identifier
- Test Environment Details

## Troubleshooting

### Common Issues

1. iPerf3 not found:
   - Verify iPerf3 installation
   - Check path provided to --iperf-path
   - Ensure iPerf3 is in system PATH

2. No jitter data:
   - Check UDP ports are not blocked
   - Verify server is running in UDP mode
   - Check network permissions

3. Report generation fails:
   - Verify write permissions in output directory
   - Check Python package installation
   - Ensure logo.png exists in img directory

### Logs

Check the log file in `network_tests/network_test_YYYYMMDD_HHMMSS.log` for detailed error information and debugging.

## Author

[Wolfgang Azevedo](https://github.com/wolfgang-azevedo) - Creator and maintainer

## Contributing

1. Fork the repository from https://github.com/wolfgang-azevedo/pynetqa
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

- iPerf3 team for their excellent network testing tool
- Matplotlib and Seaborn for visualization capabilities
- Jinja2 for HTML report generation

## Version History

* 1.0.0 (2024-10-22)
    * Initial Release by Wolfgang Azevedo
    * Complete network testing functionality
    * HTML report generation with visualizations
    * System information collection
    * Test verification features

## Project Status

Active development - Feature requests and contributions are welcome. Please check the Issues page for current development priorities and known issues.

## Support

For support and issues, please:
1. Check the troubleshooting guide above
2. Review existing Issues on GitHub
3. Open a new Issue at https://github.com/wolfgang-azevedo/pynetqa/issues with:
   - Test logs
   - System information
   - Steps to reproduce
   - Description of the expected vs actual behavior

## Repository

Project Home: https://github.com/wolfgang-azevedo/pynetqa

## Project Dependencies Installation

### iPerf3 Installation

#### Windows
1. Download iPerf3 from https://iperf.fr/iperf-download.php
2. Extract to a known location
3. Add to PATH or use --iperf-path option

#### Linux
Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install iperf3
```

CentOS/RHEL:
```bash
sudo yum install iperf3
```

### Python Dependencies
All Python dependencies are listed in requirements.txt and can be installed using pip:
```bash
pip install -r requirements.txt
```