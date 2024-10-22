import subprocess
import platform
import json
import time
import os
import sys
import argparse
from datetime import datetime
import csv
from typing import Dict, Any, List, Tuple
import socket
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

class NetworkTester:
    def __init__(self, server_ip: str, iperf_path: str = None, log_dir: str = None):
        """Initialize NetworkTester with server IP and paths."""
        self.server_ip = server_ip
        self.os_type = platform.system().lower()
        
        # Set default iperf path based on OS
        if iperf_path is None:
            self.iperf_path = 'iperf3.exe' if self.os_type == 'windows' else 'iperf3'
        else:
            self.iperf_path = iperf_path
            
        # Set up logging
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'network_tests')
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging settings."""
        log_file = os.path.join(self.log_dir, f'network_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_iperf(self) -> bool:
        """Verify iperf3 is installed and accessible."""
        try:
            subprocess.run([self.iperf_path, '--version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error("iPerf3 not found. Please install iPerf3 and ensure it's in your PATH")
            return False

    def test_connectivity(self) -> Dict[str, Any]:
        """Test basic connectivity using ping."""
        self.logger.info("Starting connectivity test...")
        
        ping_command = ['ping', self.server_ip]
        if self.os_type != 'windows':
            ping_command.extend(['-c', '10'])
        else:
            ping_command.extend(['-n', '10'])

        try:
            result = subprocess.run(ping_command, 
                                 capture_output=True, 
                                 text=True)
            
            # Parse ping results
            output = result.stdout
            packet_loss = '100%' if result.returncode else '0%'
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'packet_loss': packet_loss,
                'raw_output': output
            }
        except subprocess.SubprocessError as e:
            self.logger.error(f"Connectivity test failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def test_performance(self, duration: int = 30) -> Dict[str, Any]:
        """Run iPerf3 performance test."""
        self.logger.info("Starting performance test...")
        
        command = [
            self.iperf_path,
            '-c', self.server_ip,
            '-t', str(duration),
            '-i', '1',  # 1 second intervals
            '-J'  # JSON output
        ]

        try:
            result = subprocess.run(command, 
                                 capture_output=True, 
                                 text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'end' in data and 'streams' in data['end'] and data['end']['streams']:
                    bandwidth = data['end']['streams'][0]['receiver']['bits_per_second']
                    return {
                        'status': 'success',
                        'bandwidth': bandwidth,
                        'raw_data': data
                    }
                else:
                    return {
                        'status': 'error',
                        'error': 'Invalid performance data format'
                    }
            else:
                return {
                    'status': 'error',
                    'error': result.stderr
                }
        except Exception as e:
            self.logger.error(f"Performance test failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def test_jitter(self, duration: int = 30) -> Dict[str, Any]:
        """Test jitter using iPerf3 UDP mode with detailed debugging."""
        self.logger.info("Starting jitter test...")
        
        command = [
            self.iperf_path,
            '-c', self.server_ip,
            '-u',  # UDP mode
            '-b', '100M',  # UDP bandwidth
            '-t', str(duration),
            '-i', '1',    # 1 second intervals
            '-l', '1400', # UDP datagram size
            '-J'          # JSON output
        ]

        try:
            self.logger.info(f"Running iPerf3 command: {' '.join(command)}")
            result = subprocess.run(command, 
                                capture_output=True, 
                                text=True,
                                timeout=duration + 10)
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    
                    # Debug: Print the structure of the first interval
                    if 'intervals' in data and data['intervals']:
                        self.logger.debug(f"First interval structure: {json.dumps(data['intervals'][0], indent=2)}")
                    
                    # Extract jitter values from intervals
                    jitter_values = []
                    if 'intervals' in data:
                        for interval in data['intervals']:
                            if 'streams' in interval and interval['streams']:
                                stream = interval['streams'][0]
                                # Try to get jitter from sum
                                if 'sum' in stream and 'jitter_ms' in stream['sum']:
                                    jitter_values.append(float(stream['sum']['jitter_ms']))
                                # Try to get jitter from UDP
                                elif 'sum' in stream and 'udp' in stream['sum']:
                                    jitter = stream['sum']['udp'].get('jitter_ms')
                                    if jitter is not None:
                                        jitter_values.append(float(jitter))

                    self.logger.info(f"Number of intervals found: {len(data.get('intervals', []))}")
                    self.logger.info(f"Number of jitter values extracted: {len(jitter_values)}")

                    # Get final jitter value from end summary
                    end_jitter = None
                    if 'end' in data:
                        if 'sum_received' in data['end']:
                            end_jitter = data['end']['sum_received'].get('jitter_ms')
                        elif 'streams' in data['end'] and data['end']['streams']:
                            stream = data['end']['streams'][0]
                            if 'udp' in stream:
                                end_jitter = stream['udp'].get('jitter_ms')
                            elif 'sum' in stream:
                                end_jitter = stream['sum'].get('jitter_ms')

                    self.logger.info(f"End summary jitter: {end_jitter}")

                    # If we have no interval data but have end_jitter, create a flat line
                    if not jitter_values and end_jitter is not None:
                        jitter_values = [float(end_jitter)] * len(data.get('intervals', []))
                        self.logger.info(f"Created flat line with end jitter value: {end_jitter}")

                    return {
                        'status': 'success',
                        'jitter': float(end_jitter) if end_jitter is not None else 0,
                        'jitter_values': jitter_values,
                        'raw_data': data
                    }

                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse iPerf3 output: {str(e)}")
                    return {'status': 'error', 'error': 'Invalid JSON output'}
            else:
                self.logger.error(f"iPerf3 command failed: {result.stderr}")
                return {'status': 'error', 'error': result.stderr}

        except Exception as e:
            self.logger.error(f"Jitter test failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def test_mtu(self) -> Dict[str, Any]:
        """Test MTU sizes."""
        self.logger.info("Starting MTU test...")
        
        mtu_sizes = [1472, 1500, 1800]
        results = {}

        for size in mtu_sizes:
            command = ['ping']
            if self.os_type == 'windows':
                command.extend(['-n', '3', '-l', str(size), '-f', self.server_ip])
            else:
                command.extend(['-c', '3', '-s', str(size), '-M', 'do', self.server_ip])

            try:
                result = subprocess.run(command, 
                                     capture_output=True, 
                                     text=True)
                results[size] = {
                    'success': result.returncode == 0,
                    'output': result.stdout
                }
            except Exception as e:
                results[size] = {
                    'success': False,
                    'error': str(e)
                }

        return results

    def test_system_info(self) -> Dict[str, Any]:
        """Gather system information for the report."""
        import platform
        import uuid
        import socket

        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((self.server_ip, 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "Unable to determine"

        return {
            'os_name': platform.system(),
            'os_version': platform.version(),
            'os_platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'hostname': socket.gethostname(),
            'client_ip': local_ip,
            'python_version': platform.python_version(),
            'test_uuid': str(uuid.uuid4())
        }

    def _plot_performance(self, data: Dict, report_dir: str):
        """Create performance visualization."""
        plt.figure(figsize=(10, 6))
        intervals = data.get('intervals', [])
        
        if intervals:
            try:
                transfer_rates = []
                for interval in intervals:
                    if 'streams' in interval and interval['streams']:
                        rate = interval['streams'][0].get('bits_per_second', 0) / 1e6
                        transfer_rates.append(rate)

                if transfer_rates:
                    plt.plot(range(len(transfer_rates)), transfer_rates)
                    plt.title('Network Performance Over Time')
                    plt.xlabel('Interval (seconds)')
                    plt.ylabel('Transfer Rate (Mbps)')
                    plt.grid(True)
                    plt.savefig(os.path.join(report_dir, 'performance.png'))
            except Exception as e:
                self.logger.error(f"Error plotting performance data: {str(e)}")
                plt.text(0.5, 0.5, 'Error plotting performance data', 
                        horizontalalignment='center',
                        verticalalignment='center')
                plt.savefig(os.path.join(report_dir, 'performance.png'))
        plt.close()

    def _plot_jitter(self, data: Dict, report_dir: str):
        """Create jitter visualization with improved data handling."""
        plt.figure(figsize=(10, 6))
        plt.style.use('bmh')
        
        try:
            # Extract jitter values and ensure we have data
            jitter_values = data.get('jitter_values', [])
            final_jitter = data.get('jitter', 0)
            
            if jitter_values:
                self.logger.info(f"Plotting {len(jitter_values)} jitter values")
                timestamps = list(range(len(jitter_values)))
                
                # Create the plot
                plt.plot(timestamps, jitter_values, color='#2076D8', linewidth=2, label='Jitter')
                
                # Only add fill_between if we have varying values
                if len(set(jitter_values)) > 1:
                    plt.fill_between(timestamps, jitter_values, alpha=0.2, color='#2076D8')
                
                # Calculate statistics
                mean_jitter = sum(jitter_values) / len(jitter_values)
                max_jitter = max(jitter_values)
                min_jitter = min(jitter_values)
                
                # If all values are the same, adjust y-axis limits
                if max_jitter == min_jitter:
                    plt.ylim(max(0, mean_jitter - 0.01), mean_jitter + 0.01)
                
                stats_text = (f'Average: {mean_jitter:.3f} ms\n'
                            f'Max: {max_jitter:.3f} ms\n'
                            f'Min: {min_jitter:.3f} ms')
                
                plt.text(0.02, 0.98, stats_text,
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round',
                                facecolor='white',
                                edgecolor='#2076D8',
                                alpha=0.8))
                
                # Customize plot appearance
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('Time (seconds)', fontsize=10)
                plt.ylabel('Jitter (ms)', fontsize=10)
                plt.title('Network Jitter Over Time', fontsize=12, pad=20)
                
                plt.margins(x=0.02)
                plt.legend(loc='upper right')
                
                # Set background color
                plt.gca().set_facecolor('white')
                plt.gcf().set_facecolor('white')
                
            else:
                self.logger.warning("No jitter measurement data available")
                plt.text(0.5, 0.5, 'No jitter measurement data available\nCheck iPerf3 UDP test configuration',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=plt.gca().transAxes,
                        fontsize=12)
                plt.gca().set_axis_off()

            plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"Error plotting jitter data: {str(e)}")
            plt.clf()
            plt.text(0.5, 0.5, f'Error plotting jitter data:\n{str(e)}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    wrap=True)
            plt.gca().set_axis_off()

        # Save the plot
        try:
            output_path = os.path.join(report_dir, 'jitter.png')
            plt.savefig(output_path,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
            self.logger.info(f"Saved jitter plot to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving jitter plot: {str(e)}")
        finally:
            plt.close()

    def _plot_mtu_results(self, mtu_data: Dict, report_dir: str):
        """Create MTU test visualization with proper data type handling."""
        plt.figure(figsize=(10, 6))
        plt.style.use('bmh')
        
        try:
            # Convert keys to integers and create sorted list of MTU sizes and results
            mtu_items = []
            for size_str, data in mtu_data.items():
                try:
                    size = int(size_str)
                    success = bool(data.get('success', False))
                    mtu_items.append((size, success))
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Error converting MTU size {size_str}: {str(e)}")
                    continue
            
            # Sort by MTU size
            mtu_items.sort(key=lambda x: x[0])
            
            if mtu_items:
                sizes = [item[0] for item in mtu_items]
                success_values = [1 if item[1] else 0 for item in mtu_items]
                
                # Create bar plot
                bars = plt.bar(range(len(sizes)), success_values, color='#2076D8', alpha=0.7)
                
                # Customize the plot
                plt.title('MTU Test Results', fontsize=12, pad=20)
                plt.xlabel('MTU Size (bytes)', fontsize=10)
                plt.ylabel('Success (1) / Failure (0)', fontsize=10)
                
                # Set x-axis ticks to show MTU sizes
                plt.xticks(range(len(sizes)), [str(size) for size in sizes], rotation=0)
                
                # Add grid on y-axis only
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Set background color
                plt.gca().set_facecolor('white')
                plt.gcf().set_facecolor('white')
                
                # Add success/failure labels on top of bars
                for i, success in enumerate(success_values):
                    label = 'Success' if success else 'Failed'
                    color = 'green' if success else 'red'
                    plt.text(i, success + 0.05,
                            label,
                            ha='center',
                            va='bottom',
                            color=color,
                            fontsize=9)
                
                plt.ylim(0, 1.2)  # Set y-axis limit to make room for labels
                plt.tight_layout()
                
            else:
                plt.text(0.5, 0.5, 'No MTU data available',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=plt.gca().transAxes,
                        fontsize=12)
                plt.gca().set_axis_off()
                
        except Exception as e:
            self.logger.error(f"Error plotting MTU data: {str(e)}")
            plt.clf()
            plt.text(0.5, 0.5, f'Error plotting MTU data: {str(e)}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    wrap=True)
            plt.gca().set_axis_off()

        try:
            plt.savefig(os.path.join(report_dir, 'mtu.png'),
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        except Exception as e:
            self.logger.error(f"Error saving MTU plot: {str(e)}")
        finally:
            plt.close()

    def generate_report(self, results: Dict[str, Any]) -> Tuple[str, str]:
        """Generate enhanced report with system info and test verification."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add system information
        results['system_info'] = self.test_system_info()
        
        # Add report paths
        report_dir = os.path.join(self.log_dir, f'report_{timestamp}')
        results['report_info'] = {
            'timestamp': timestamp,
            'report_dir': report_dir,
            'json_path': os.path.join(report_dir, 'results.json'),
            'log_path': os.path.join(self.log_dir, f'network_test_{timestamp}.log'),
            'test_id': results['system_info']['test_uuid']
        }
        
        # Create directories and generate report as before
        os.makedirs(report_dir, exist_ok=True)
        
        # Create img directory in report folder
        img_dir = os.path.join(report_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        
        # Copy logo to report directory
        try:
            logo_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img', 'logo.png')
            logo_dst = os.path.join(img_dir, 'logo.png')
            if os.path.exists(logo_src):
                import shutil
                shutil.copy2(logo_src, logo_dst)
            else:
                self.logger.warning("Logo file not found at: " + logo_src)
        except Exception as e:
            self.logger.error(f"Error copying logo: {str(e)}")
        
        # Save raw JSON results
        json_path = os.path.join(report_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate visualizations
        if results['tests']['performance']['status'] == 'success':
            self._plot_performance(results['tests']['performance']['raw_data'], report_dir)
        if results['tests']['jitter']['status'] == 'success':
            self._plot_jitter(results['tests']['jitter'], report_dir)
        self._plot_mtu_results(results['tests']['mtu'], report_dir)

        # Create HTML report
        html_path = os.path.join(report_dir, 'report.html')
        html_content = self._generate_html_report(results, timestamp)
        
        with open(html_path, 'w') as f:
            f.write(html_content)

        return json_path, html_path

    def _generate_html_report(self, results: Dict[str, Any], timestamp: str) -> str:
        """Generate HTML report using template with enhanced features."""
        from jinja2 import Environment, BaseLoader

        # Create Jinja2 environment with custom filters
        env = Environment(loader=BaseLoader())
        
        def format_datetime(value):
            """Format datetime for display."""
            try:
                if isinstance(value, str):
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(value, "%Y%m%d_%H%M%S")
                return dt.strftime("%A, %B %d, %Y %I:%M:%S %p")
            except Exception as e:
                self.logger.error(f"Error formatting datetime: {str(e)}")
                return value

        def get_max_working_mtu(mtu_results):
            """Get the maximum working MTU size."""
            working_mtus = [int(size) for size, data in mtu_results.items() 
                        if data.get('success', False)]
            return max(working_mtus) if working_mtus else None

        def get_working_mtus(mtu_results):
            """Get list of working MTU sizes."""
            return sorted([int(size) for size, data in mtu_results.items() 
                        if data.get('success', False)])

        def format_bandwidth(bits_per_second):
            """Format bandwidth in appropriate units."""
            if bits_per_second >= 1e9:
                return f"{bits_per_second/1e9:.2f} Gbps"
            elif bits_per_second >= 1e6:
                return f"{bits_per_second/1e6:.2f} Mbps"
            elif bits_per_second >= 1e3:
                return f"{bits_per_second/1e3:.2f} Kbps"
            return f"{bits_per_second:.2f} bps"

        # Add custom filters and functions
        env.filters['format_datetime'] = format_datetime
        env.filters['format_bandwidth'] = format_bandwidth
        env.globals.update({
            'current_year': datetime.now().year,
            'get_max_working_mtu': get_max_working_mtu,
            'get_working_mtus': get_working_mtus
        })

        template = env.from_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Network Test Report - {{ timestamp|format_datetime }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    position: relative;
                    min-height: 100vh;
                    padding-bottom: 100px;
                }
                .header {
                    background-color: #ffffff;
                    padding: 20px;
                    border-bottom: 2px solid #eee;
                    border-radius: 8px 8px 0 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                }
                .header-left {
                    flex: 1;
                }
                .header-right {
                    flex: 0 0 auto;
                    padding-left: 20px;
                }
                .logo {
                    max-height: 60px;
                    width: auto;
                }
                h1 {
                    color: #2c3e50;
                    margin: 0 0 10px 0;
                    font-size: 24px;
                }
                .info-line {
                    color: #666;
                    font-size: 14px;
                    margin: 5px 0;
                }
                .test-id {
                    font-family: monospace;
                    color: #666;
                    font-size: 12px;
                    margin-top: 10px;
                }
                .section {
                    margin: 20px;
                    padding: 20px;
                    background-color: white;
                    border: 1px solid #eee;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                }
                .section h2 {
                    color: #2c3e50;
                    margin-top: 0;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #eee;
                }
                .success { 
                    color: #27ae60;
                    font-weight: bold;
                }
                .error { 
                    color: #e74c3c;
                    font-weight: bold;
                }
                .warning { 
                    color: #f39c12;
                    font-weight: bold;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                    background-color: white;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #eee;
                }
                th {
                    background-color: #f8f9fa;
                    color: #2c3e50;
                    font-weight: bold;
                }
                tr:hover {
                    background-color: #f8f9fa;
                }
                .visualization {
                    margin: 20px 0;
                    text-align: center;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                }
                .visualization img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                }
                .metric-value {
                    font-weight: bold;
                    color: #2c3e50;
                }
                .system-info-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .info-item {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #2076D8;
                }
                .info-item h3 {
                    margin-top: 0;
                    color: #2c3e50;
                    font-size: 16px;
                }
                .mtu-summary {
                    margin-top: 10px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    border-left: 4px solid #2076D8;
                }
                .mtu-tag {
                    display: inline-block;
                    padding: 2px 8px;
                    margin: 2px;
                    border-radius: 3px;
                    background-color: #e7f3ff;
                    color: #2076D8;
                    font-size: 0.9em;
                }
                .nested-table {
                    margin: 10px 0;
                    font-size: 0.9em;
                    width: 100%;
                    border-radius: 5px;
                    overflow: hidden;
                }
                .nested-table th,
                .nested-table td {
                    padding: 8px;
                    text-align: left;
                }
                code {
                    background-color: #f8f9fa;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: monospace;
                    font-size: 0.9em;
                }
                .footer {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    font-size: 0.9em;
                    position: absolute;
                    bottom: 0;
                    width: 100%;
                    border-radius: 0 0 8px 8px;
                    box-sizing: border-box;
                }
                .footer p {
                    margin: 5px 0;
                    font-size: 12px;
                }
                @media print {
                    body {
                        background-color: white;
                    }
                    .container {
                        box-shadow: none;
                    }
                    .visualization img {
                        max-width: 500px;
                    }
                    .footer {
                        position: fixed;
                        bottom: 0;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="header-left">
                        <h1>Network Test Report</h1>
                        <p class="info-line">Generated: {{ timestamp|format_datetime }}</p>
                        <p class="info-line">Server IP: {{ results.server_ip }}</p>
                        <p class="info-line">Client IP: {{ results.system_info.client_ip }}</p>
                        <p class="test-id">Test ID: {{ results.report_info.test_id }}</p>
                    </div>
                    <div class="header-right">
                        <img src="img/logo.png" alt="Company Logo" class="logo">
                    </div>
                </div>

                <div class="section">
                    <h2>System Information</h2>
                    <div class="system-info-grid">
                        <div class="info-item">
                            <h3>Operating System</h3>
                            <table>
                                <tr><td>Name:</td><td>{{ results.system_info.os_name }}</td></tr>
                                <tr><td>Version:</td><td>{{ results.system_info.os_version }}</td></tr>
                                <tr><td>Platform:</td><td>{{ results.system_info.os_platform }}</td></tr>
                            </table>
                        </div>
                        <div class="info-item">
                            <h3>Network</h3>
                            <table>
                                <tr><td>Hostname:</td><td>{{ results.system_info.hostname }}</td></tr>
                                <tr><td>Client IP:</td><td>{{ results.system_info.client_ip }}</td></tr>
                                <tr><td>Server IP:</td><td>{{ results.server_ip }}</td></tr>
                            </table>
                        </div>
                        <div class="info-item">
                            <h3>Test Information</h3>
                            <table>
                                <tr><td>Test ID:</td><td>{{ results.report_info.test_id }}</td></tr>
                                <tr><td>Python Version:</td><td>{{ results.system_info.python_version }}</td></tr>
                                <tr><td>Machine:</td><td>{{ results.system_info.machine }}</td></tr>
                            </table>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>Test Results</h2>
                    <table>
                        <tr>
                            <th>Test</th>
                            <th>Status</th>
                            <th>Result</th>
                        </tr>
                        <tr>
                            <td>Connectivity</td>
                            <td class="{{ results.tests.connectivity.status }}">
                                {{ results.tests.connectivity.status|title }}
                            </td>
                            <td>
                                {% if results.tests.connectivity.status == 'success' %}
                                    <span class="metric-value">Packet Loss: {{ results.tests.connectivity.packet_loss }}</span>
                                {% else %}
                                    <span class="error">{{ results.tests.connectivity.error }}</span>
                                {% endif %}
                            </td>
                        </tr>
                        <tr>
                            <td>Performance</td>
                            <td class="{{ results.tests.performance.status }}">
                                {{ results.tests.performance.status|title }}
                            </td>
                            <td>
                                {% if results.tests.performance.status == 'success' %}
                                    <span class="metric-value">{{ results.tests.performance.bandwidth|format_bandwidth }}</span>
                                {% else %}
                                    <span class="error">{{ results.tests.performance.error }}</span>
                                {% endif %}
                            </td>
                        </tr>
                        <tr>
                            <td>Jitter</td>
                            <td class="{{ results.tests.jitter.status }}">
                                {{ results.tests.jitter.status|title }}
                            </td>
                            <td>
                                {% if results.tests.jitter.status == 'success' %}
                                    <span class="metric-value">{{ results.tests.jitter.jitter|round(3) }} ms</span>
                                {% else %}
                                    <span class="error">{{ results.tests.jitter.error }}</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% if results.tests.mtu.items()|length > 0 %}
                        <tr>
                            <td>MTU Test</td>
                            <td class="{{ 'success' if get_working_mtus(results.tests.mtu)|length > 0 else 'error' }}">
                                {{ 'Success' if get_working_mtus(results.tests.mtu)|length > 0 else 'Failed' }}
                            </td>
                            <td>
                                <div class="mtu-results">
                                    <span class="metric-value">Maximum supported MTU: {{ get_max_working_mtu(results.tests.mtu) }} bytes</span>
                                    <table class="nested-table">
                                        <tr>
                                            <th>MTU Size</th>
                                            <th>Status</th>
                                            <th>Result</th>
                                        </tr>
                                        {% for size in results.tests.mtu.keys()|sort %}
                                        <tr>
                                            <td>{{ size }} bytes</td>
                                            <td class="{{ 'success' if results.tests.mtu[size].success else 'error' }}">
                                                {{ 'Success' if results.tests.mtu[size].success else 'Failed' }}
                                            </td>
                                            <td>{{ 'Supported' if results.tests.mtu[size].success else 'Not Supported' }}</td>
                                        </tr>
                                        {% endfor %}
                                    </table>
                                </div>
                            </td>
                        </tr>
                        {% endif %}
                    </table>
                </div>

                {% if results.tests.performance.status == 'success' %}
                <div class="section">
                    <h2>Performance Analysis</h2>
                    <div class="visualization">
                        <img src="performance.png" alt="Performance Graph">
                    </div>
                </div>
                {% endif %}

                {% if results.tests.jitter.status == 'success' %}
                <div class="section">
                    <h2>Jitter Analysis</h2>
                    <div class="visualization">
                        <img src="jitter.png" alt="Jitter Graph">
                    </div>
                </div>
                {% endif %}

                <div class="section">
                    <h2>MTU Test Results</h2>
                    <div class="visualization">
                        <img src="mtu.png" alt="MTU Test Results">
                    </div>
                </div>

                <div class="section">
                    <h2>Report Information</h2>
                    <table>
                        <tr>
                            <th>Item</th>
                            <th>Path</th>
                        </tr>
                        <tr>
                            <td>Report Directory</td>
                            <td><code>{{ results.report_info.report_dir }}</code></td>
                        </tr>
                        <tr>
                            <td>JSON Report</td>
                            <td><code>{{ results.report_info.json_path }}</code></td>
                        </tr>
                        <tr>
                            <td>Log File</td>
                            <td><code>{{ results.report_info.log_path }}</code></td>
                        </tr>
                    </table>
                </div>

                <div class="footer">
                    <p>© {{ current_year }} Network Testing Tool</p>
                    <p>This program is free software: you can redistribute it and/or modify
                    it under the terms of the GNU General Public License as published by
                    the Free Software Foundation, either version 3 of the License, or
                    (at your option) any later version.</p>
                    <p>Report generated on: {{ timestamp|format_datetime }}</p>
                    <p>Test ID: {{ results.report_info.test_id }}</p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        try:
            return template.render(results=results, timestamp=timestamp)
        except Exception as e:
            self.logger.error(f"Error rendering template: {str(e)}")
            # Fallback to basic error template
            error_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error Report</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        padding: 20px; 
                        background-color: #f5f5f5;
                    }
                    .error { 
                        color: #e74c3c;
                        font-weight: bold;
                    }
                    .container {
                        background-color: white;
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }
                    pre { 
                        background: #f8f9fa; 
                        padding: 10px; 
                        border-radius: 5px;
                        overflow-x: auto;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Error Generating Report</h1>
                    <p class="error">An error occurred while generating the report: {{ error }}</p>
                    <h2>Raw Data:</h2>
                    <pre>{{ data }}</pre>
                </div>
            </body>
            </html>
            """
            error_env = Environment(loader=BaseLoader())
            template = error_env.from_string(error_template)
            return template.render(
                error=str(e),
                data=json.dumps(results, indent=2)
            )
   
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all network tests with enhanced jitter testing."""
        self.logger.info("Starting comprehensive network testing...")
        
        if not self.check_iperf():
            return {'status': 'error', 'message': 'iPerf3 not found'}

        results = {
            'timestamp': datetime.now().isoformat(),
            'server_ip': self.server_ip,
            'tests': {
                'connectivity': self.test_connectivity(),
                'performance': self.test_performance(),
                'jitter': self.test_jitter(),  # Use new comprehensive test
                'mtu': self.test_mtu()
            }
        }

        json_path, html_path = self.generate_report(results)
        self.logger.info(f"Testing complete. Reports saved to:")
        self.logger.info(f"JSON Report: {json_path}")
        self.logger.info(f"HTML Report: {html_path}")

        return results

def main():
    parser = argparse.ArgumentParser(description='Network Testing Tool')
    parser.add_argument('--server', '-s', required=True, help='iPerf server IP address')
    parser.add_argument('--iperf-path', help='Path to iPerf3 executable')
    parser.add_argument('--log-dir', help='Directory for log files')
    parser.add_argument('--test', choices=['all', 'connectivity', 'performance', 'jitter', 'mtu'],
                      default='all', help='Specific test to run')
    
    args = parser.parse_args()

    tester = NetworkTester(args.server, args.iperf_path, args.log_dir)

    if args.test == 'all':
        results = tester.run_all_tests()
    else:
        test_method = getattr(tester, f'test_{args.test}')
        results = test_method()

    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()