import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from pynput.mouse import Controller
import os
import csv
import glob
import traceback
from datetime import datetime, timedelta
import re 

class WAHISWebScraper:
    def __init__(self, output_path):
        """Initialize the scraper with the website URL."""
        #output_path = output_path.replace("/mapped_combined_file.csv", "")
        output_path = os.path.abspath(output_path)
        self.chrome_options = Options()
        prefs = {
            "profile.default_content_settings.popups": 0,
            "download.default_directory": output_path,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False,
            "profile.default_content_setting_values.automatic_downloads": 1 
        }
        self.chrome_options.add_experimental_option("prefs", prefs)
        #self.chrome_options.add_argument("--headless=new")
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.mouse = Controller()
        self.page_count = 2
        self.output_path = output_path

    def clear_browser_state(self):
        """Clear browser state to reduce memory usage."""
        try:
            self.driver.delete_all_cookies()

            current_url = self.driver.execute_script("return window.location.href;")
            if not current_url.startswith("data:"):
                self.driver.execute_script("window.localStorage.clear();")
                self.driver.execute_script("window.sessionStorage.clear();")
            else:
                print("Skipping localStorage clear due to 'data:' URL.")

        except Exception as e:
            print(f"Error clearing browser state: {e}")

    def restart_browser(self):
        """Restart the browser to clear memory."""
        #current_url = self.driver.current_url
        self.driver.quit()
        self.driver = webdriver.Chrome(options=self.chrome_options)
        #self.driver.get(current_url)

    def open_website(self, website):
        """Open the website in the browser."""
        self.clear_browser_state()
        self.driver.get(website)
        try:
            WebDriverWait(self.driver, 20).until(
                lambda driver: driver.find_elements(By.TAG_NAME, 'app-custom-table') or
                            driver.find_elements(By.CSS_SELECTOR, 'button.mat-focus-indicator.mat-flat-button.mat-button-base')
            )
            
            if self.driver.find_elements(By.TAG_NAME, 'app-custom-table'):
                print("Found table as 'app-custom-table'")
            elif self.driver.find_elements(By.CSS_SELECTOR, 'button.mat-focus-indicator.mat-flat-button.mat-button-base'):
                print("Found clickable button")
            time.sleep(1)
        except Exception as e:
            print(f"Table or button did not load: {e}")

    def set_items_per_page(self, website):
        """Set the number of items displayed per page to 100."""
        page_dropdown = WebDriverWait(self.driver, 15).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.mat-select-arrow-wrapper'))
        )
        page_dropdown.click()

        hundred_option = WebDriverWait(self.driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, '//mat-option[.//span[contains(text(), "100")]]'))
        )
        hundred_option.click()
        self.open_website(website)

    def get_total_pages(self):
        """Calculate the total number of pages based on the range label."""
        try:
            page_label = WebDriverWait(self.driver, 15).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, 'div.mat-paginator-range-label'))
            )
            total_items = int(page_label.text.split()[-1])
            self.page_count = (total_items // 100) + 1
            print('Number of pages:', self.page_count)
        except Exception as e:
            print(f"Failed to calculate total pages: {e}")
            self.page_count = 0

    def download_data(self):
        """Download data for the current page and navigate to the next page."""
        for i in range(self.page_count):
            print(f"Processing page {i + 1} of {self.page_count}")
            # Wait until the button is present
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'button.setting-button.mat-button'))
            )
            self.driver.execute_script("window.scroll(0, 0);")

            # Click the settings button to download
            settings_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.setting-button.mat-button'))
            )
            settings_button.click()

            # Click the export button
            export_button = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//button[.//span[contains(text(), "Export")]]')
                )
            )
            export_button.click()

            if i + 1 < self.page_count:
                # Navigate to the next page
                next_button = WebDriverWait(self.driver, 15).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.mat-paginator-navigation-next.mat-icon-button'))
                )
                next_button.click()

    def close_cookie_popup(self):
        """Close the cookie preference popup."""
        try:
            cookies = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.close-cookie-pref'))
            )
            self.driver.execute_script("arguments[0].click();", cookies)
        except Exception as e:
            print(f"No cookie popup: {e}")

    def click_report_button(self):
        """Click on the main report button."""
        try:
            report_button = WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.mat-focus-indicator.mat-flat-button.mat-button-base'))
            )
            self.driver.execute_script("arguments[0].click();", report_button)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h6'))
            )
            print("Clicked report button")
        except Exception as e:
            print(f"Report button not found: {e}")

    def extract_general_info(self):
        """Read the information in the general panel section."""
        try:
            read_gen_info = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'app-event-information-review'))
            )
            text_gen_info = read_gen_info.text
            list_gen_info = text_gen_info.split('\n')
            col_gen, val_gen = [], []
            for i in range(1, len(list_gen_info), 2):
                col_gen.append(list_gen_info[i - 1])
                val_gen.append(list_gen_info[i])
            return col_gen, val_gen
        except Exception as e:
            print(f"Error extracting general info: {e}")
            return [], []
    

    def expand_outbreak_sections(self, ignored_outbreaks, outbreaks_to_open):
        """Click open all the outbreak sections."""
        outbreak_list = []

        headers = WebDriverWait(self.driver, 15).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, 'mat-expansion-panel-header.custom-expansion-panel-header')
            )
        )
        print("expanding oubreak section")

        opened_outbreaks = 0
        for header in headers:
            if opened_outbreaks == outbreaks_to_open:
                break
            # Get the text content of the dropdown
            text_span = header.find_element(By.CSS_SELECTOR, 'span[style*="margin"]')
            text_drops = text_span.text
            if text_drops in ignored_outbreaks or not text_drops:
                continue

            if text_drops.startswith('OB') or text_drops[0].isdigit():
                outbreak_list.append(text_drops)

                try:
                    # Wait for the header to be clickable
                    WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(header)
                    )
                    
                    # Click the header
                    self.driver.execute_script("arguments[0].click();", header)

                    # Wait for the dropdown to fully expand
                    WebDriverWait(self.driver, 5).until(
                        lambda d: header.get_attribute("aria-expanded") == "true"
                    )

                    print(f"Dropdown expanded: {text_drops}")
                    opened_outbreaks += 1

                except Exception as e:
                    print(f"Error expanding dropdown: {e}")
                    traceback.print_exc() 
        
        # Wait until it reads in all the outbreak dropdowns
        read_ob_info = WebDriverWait(self.driver, 15).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, 'app-outbreak-information-review')) == opened_outbreaks,
            message=f"Expected {outbreaks_to_open} elements, but condition not met."
        )
        # Wait for *at least some* outbreak info elements to appear (instead of all)
        read_ob_info = self.driver.find_elements(By.CSS_SELECTOR, 'app-outbreak-information-review')

        read_ob_info = self.driver.find_elements(By.CSS_SELECTOR, 'app-outbreak-information-review')
        extracted_ob_info = [info.text for info in read_ob_info]
        read_quant_info = WebDriverWait(self.driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'app-quantitative-data-review'))
        )
        extracted_quant_info = [info.text for info in read_quant_info[1:]]
        return outbreak_list, extracted_ob_info, extracted_quant_info

    

    def extract_outbreak_data(self, read_ob_info, read_quant_info, i):
        text_ob_info = read_ob_info[i]
        list_ob_info = text_ob_info.split('\n')
        #print(f"list_ob_info: {list_ob_info}")
        
        col_quant = ['Species', 'New Susceptible', 'New Cases', 'New Deaths', 'New Killed/Disposed', 'New killed for commercial use', 'New Vaccinated',
             'Total Susceptible', 'Total Cases', 'Total Deaths', 'Total Killed/Disposed', 'Total killed for commercial use', 'Total Vaccinated']
        col_ob, val_ob, val_quant = [], [], []
        if '(Approximate location)' in list_ob_info:
            list_ob_info.remove('(Approximate location)')

        for j in range (1, len(list_ob_info), 2):
            col_ob.append(list_ob_info[j - 1])
            val_ob.append(list_ob_info[j])

        text_quant_info = read_quant_info[i]
        list_quant_info = text_quant_info.split("\n")
        #print(f"list_quant_info: {list_quant_info}")
        #print((list_quant_info), len(list_quant_info))
        var = []
        if (len(list_quant_info) > 7):
            val_quant.append(list_quant_info[7])
            val_quant.extend(list_quant_info[-13:-7]) # 9:15
            val_quant.extend(list_quant_info[-6:]) # 16:22
            if(len(list_quant_info) > 23):                        # variable number of species in the table 
                num_elements = len(list_quant_info[23:])
                var = ['a' + str(i + 1) for i in range(num_elements)]
                val_quant.extend(list_quant_info[23:]) 
        else:
            col_quant, val_quant = [], []

        #print(f"col_ob: {col_ob}")
        #print(f"val_ob: {val_ob}")
        #print(f"col_quant: {col_quant}")
        #print(f"val_quant: {val_quant}")
        return col_ob, val_ob, col_quant, val_quant, var

    def create_outbreak_df(self, outbreak_list, col, val, col_gen, val_gen, col_ob, val_ob, col_quant, val_quant, var, i):
        col.append('Outbreak_number')
        val.append(outbreak_list[i])
        # One list of: general info + dropdown info + species table 
        col = col_gen + col_ob + col_quant + var                        
        val = val_gen + val_ob + val_quant
        print('col_gen=', len(col_gen), '\n val_gen', len(val_gen), 'col_ob=', len(col_ob), '\n val_ob', len(val_ob), 'col_quant=', len(col_quant), '\n val_quant', len(val_quant),'\n col', len(col), 'val=', len(val), 'var=', len(var))
        df = pd.DataFrame(columns = col)
        #val = [item.replace(',', '') if isinstance(item, str) else item for item in val]
        val = [item.replace(',', '') if isinstance(item, str) and idx != 23 else item for idx, item in enumerate(val)]
        df.loc[len(df)] = val
        return df
    
    def save_to_csv(self, data, output_file):
        """Save the collected data to a CSV file."""
        df = pd.DataFrame(data)
        if os.path.exists(output_file):
            df.to_csv(output_file, mode='a', header=False)
        else:
            df.to_csv(output_file, mode='w', header=True)
    
    def combine_to_csv(self, source_folder, destination_file):
        file_pattern = os.path.join(source_folder, "Event-list-*.csv")
        csv_files = sorted(glob.glob(file_pattern))

        if not csv_files:
            print("No files matching the pattern found.")
            return

        combined_df = pd.concat([pd.read_csv(file, sep=';') for file in csv_files], ignore_index=True)
        combined_df.to_csv(destination_file, index=False)
    
    def remove_event_list_files(self, source_folder):
        file_pattern = os.path.join(source_folder, "Event-list-*.csv")
        csv_files = glob.glob(file_pattern)

        for file in csv_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

    def extract_data_page(self, event_id, ignored_outbreaks, first_run, new_page):
        website =('https://wahis.woah.org/#/in-event/'+str(event_id)+'/dashboard')
        self.open_website(website)

        if first_run:
            # Close the cookie preference popup
            self.close_cookie_popup()
        # Click on the main report button
        self.click_report_button()
        # Read the information in the general panel section
        if new_page:
            col_gen, val_gen = self.extract_general_info()
        else:
            col_gen, val_gen = '', ''

        # Click open all the outbreak sections
        outbreak_list, read_ob_info, read_quant_info = self.expand_outbreak_sections(ignored_outbreaks, 200)
        return outbreak_list, read_ob_info, read_quant_info, col_gen, val_gen
        """
        self.open_website(website)
        self.click_report_button()
        outbreak_list2, read_ob_info2, read_quant_info2 = self.expand_outbreak_sections(outbreak_list, 2)
        print("=======testing joined==================")
        joined_ob_info = read_ob_info + read_ob_info2
        joined_quant_info = read_quant_info + read_quant_info2
        print(len(joined_ob_info))
        for i in range(len(joined_ob_info)):
            col_ob, val_ob, col_quant, val_quant, var = self.extract_outbreak_data(joined_ob_info, joined_quant_info, i)
        print("=======endtesting joined==================")

        self.open_website(website)
        self.click_report_button()
        outbreak_list3, read_ob_info3, read_quant_info3 = self.expand_outbreak_sections([], 4)
        """

    def extract_data(self, file_name, destination_file):
        df = pd.read_csv(file_name)
        allowed_diseases = [
            "high pathogenicity avian influenza viruses (poultry) (inf. with)",
            "influenza a viruses of high pathogenicity (inf. with) (non-poultry including wild birds) (2017-)",
            "low pathogenicity avian influenza viruses transmissible to humans (inf. with) (2022-)"
        ]
        # Filter out diseases and events that are older than a week
        #df = df[df['disease'].str.lower().isin(allowed_diseases)]
        eventId = df.loc[:,"eventId"]
        #one_week_ago = datetime.now() - timedelta(weeks=1)
        #df = df[pd.to_datetime(df['submissionDate'], errors='coerce') >= one_week_ago]

        col = []
        val = []
        var = []

        if os.path.exists(destination_file):
            df_processed = df = pd.read_csv(destination_file, header=None, usecols=[4])
            processed_event_ids = set(df_processed[4])
        else:
            processed_event_ids = []

        first_run = True
        total_outbreaks_fetched = 0
        for event_id in eventId:
            if str(event_id) in processed_event_ids:
                print(f"skipping {event_id}")
                continue

            opened_outbreaks = []
            all_ob_info = None
            all_quant_info = None
            final_col_gen = None
            final_val_gen = None
            new_page = True
            while True:
                outbreak_list, read_ob_info, read_quant_info, col_gen, val_gen = self.extract_data_page(event_id, opened_outbreaks, first_run, new_page)
                first_run = False
                new_page = False
                if not outbreak_list:
                    break
                all_ob_info = read_ob_info if not all_ob_info else all_ob_info + read_ob_info
                all_quant_info = read_quant_info if not all_quant_info else all_quant_info + read_quant_info
                final_col_gen = col_gen if not final_col_gen else final_col_gen
                final_val_gen = val_gen if not final_val_gen else final_val_gen
                opened_outbreaks = opened_outbreaks + outbreak_list
                total_outbreaks_fetched = total_outbreaks_fetched + len(outbreak_list)
                print(f"Processing {len(opened_outbreaks)} outbreaks")
                if total_outbreaks_fetched > 1000:
                    print("Restarting browser")
                    self.restart_browser()
                    first_run = True
                    total_outbreaks_fetched = 0

            if not all_ob_info:
                continue
            for i in range(len(all_ob_info)):
                col_ob, val_ob, col_quant, val_quant, var = self.extract_outbreak_data(all_ob_info, all_quant_info, i)
                df = self.create_outbreak_df(opened_outbreaks, col, val, final_col_gen, final_val_gen, col_ob, val_ob, col_quant, val_quant, var, i)
                self.save_to_csv(df, destination_file)

    def extract_bad_lines_with_csv(self, source_file, destination_file, expected_fields=41, delimiter=',', encoding='utf-8'):
        with open(source_file, 'r', encoding=encoding) as file:
            reader = csv.reader(file, delimiter=delimiter)
            header = next(reader)
            bad_lines = [f"{','.join(header)}\n"]
            
            line_number = 0
            for line_number, row in enumerate(reader, start=2):  # Start from line 2 since header is line 1
                if len(row) > expected_fields:
                    bad_lines.append(f"Line {line_number}: {','.join(row)}\n")

        # Check if there are any bad lines and write them to the output file
        if bad_lines:
            with open(destination_file, 'w', encoding=encoding) as outfile:
                outfile.writelines(bad_lines)
        else:
            print("No bad lines found.")
            
    def process_row(self, row, is_header=False):
        # Directly return the header as a DataFrame if it's the header row
        if is_header:
            return pd.DataFrame([row[0].split(',')])
        
        # Split the row into parts for processing
        parts = row[0].split(',')
        
        # Identify and remove any instance of 'ALL SPECIES' and everything after it
        if 'All species' in parts:
            all_species_index = parts.index('All species')
            parts = parts[:all_species_index]
        
        # Initialize the list to collect processed parts
        processed_rows = []

        # Copy columns 1-29 to new rows (index 28 because it's zero-based)
        base_row = parts[:41]
        processed_rows.append(base_row)

        # Process subsequent segments of 13 columns after col 41
        extra_parts = parts[42:]
        
        # Keywords to remove, treated case-sensitively
        keywords = {'NEW', 'TOTAL', 'Wild', 'Captive', 'All species'}
        
        # Remove specific keywords from the extra parts (case-sensitive)
        #extra_parts = [item for item in extra_parts if item not in keywords]
        extra_parts = [
            item for item in extra_parts 
            if item == '-'
            or re.fullmatch(r'\d+(\.\d+)?', item)
            or '(DOMESTIC)' in item
            or '(WILD)' in item 
        ]
        
        # print('extra_parts cleaned', extra_parts, len(extra_parts))
        # Create new rows based on remaining extra parts in chunks of 13
        while len(extra_parts) >= 13:
            new_row = base_row[0:29].copy() + extra_parts[:13]
            processed_rows.append(new_row)
            extra_parts = extra_parts[13:]
            print(new_row)

        return pd.DataFrame(processed_rows)

    def create_cleaned_csv(self, bad_file, destination_file):
        df = pd.read_csv(bad_file, header=None, sep='\t', engine='python')
        header_df = self.process_row(df.iloc[0], is_header=True)

        # Process each line (excluding the header) and collect all new rows
        processed_dataframes = [self.process_row(row) for index, row in df.iterrows() if index > 0]

        # Concatenate all new rows into a single DataFrame
        new_df = pd.concat([header_df] + processed_dataframes, ignore_index=True)
        new_df.to_csv(destination_file, index=False, header=None)

    def join_long_lat(self, source_file, destination_file):
        # Assuming 'Latitude' and 'Longitude' are the correct column names
        # Create the 'Lat_Long' column right before the 'Latitude' column
        df = pd.read_csv(source_file)
        df.insert(loc=df.columns.get_loc('Latitude'),
                column='Latitude, Longitude',
                value=df['Latitude'].astype(str) + ',' + df[' Longitude'].astype(str))

        # Now drop the original 'Latitude' and 'Longitude' columns
        df.drop(['Latitude', ' Longitude'], axis=1, inplace=True)
        df.to_csv(destination_file, index=False)

    def remove_bad_lines_from_csv(self, source_file, destination_file, expected_fields=41, delimiter=',', encoding='utf-8'):
        # Read all lines from the source file and collect lines with the correct number of fields
        with open(source_file, 'r', encoding=encoding) as infile:
            reader = csv.reader(infile, delimiter=delimiter)
            # Initialize a list to hold the good lines, starting with the header
            good_lines = [next(reader)[:expected_fields]]  # Read and store the header
            
            # Go through the rest of the lines and store only those with the correct number of fields
            for row in reader:
                if len(row) <= expected_fields:
                    good_lines.append(row)
        
        # Write the good lines back to the source file
        with open(destination_file, 'w', newline='', encoding=encoding) as outfile:
            writer = csv.writer(outfile, delimiter=delimiter)
            writer.writerows(good_lines)

    def combine_final(self, source_file_1, source_file_2, destination_file):
        df1 = pd.read_csv(source_file_1)
        df2 = pd.read_csv(source_file_2)
        combined_df = pd.concat([df1, df2], ignore_index=True)
        combined_df.to_csv(destination_file, index=False)

    def map_event_id(self, source_file_1, source_file_2, destination_file):
        wahis_event_id = pd.read_csv(source_file_1)
        combined_file = pd.read_csv(source_file_2, low_memory=False)
        wahis_event_id.rename(columns={'eventId': 'EVENT ID'}, inplace=True)
        merged_file = pd.merge(combined_file, wahis_event_id[['EVENT ID', 'country']], on='EVENT ID', how='left')
        merged_file = pd.merge(merged_file, wahis_event_id[['EVENT ID', 'reportNumber']], on='EVENT ID', how='left')
        merged_file.to_csv(destination_file, index=False) 
    
    def run(self):
        os.makedirs('./input', exist_ok=True)
        os.makedirs('./output', exist_ok=True)

        website = "https://wahis.woah.org/#/event-management"
        input_folder = "./input"
        output_folder = "./output"
        downloaded_events_file = os.path.join(input_folder, "Wahis_event_id.csv")

        if not os.path.exists(downloaded_events_file):
            self.remove_event_list_files(input_folder)
            self.open_website(website)
            self.close_cookie_popup()
            self.set_items_per_page(website)
            self.get_total_pages()
            self.download_data()

            # Check downloaded event files
            event_list_files = glob.glob(os.path.join(input_folder, "Event-list-*.csv"))
            print("Downloaded event files:", event_list_files)
            if not event_list_files:
                raise FileNotFoundError("No Event-list CSVs were downloaded into ./input.")

            # Combine CSVs first
            self.combine_to_csv(input_folder, downloaded_events_file)

            # ✅ NOW filter only influenza rows
            df_events = pd.read_csv(downloaded_events_file)
            df_events = df_events[df_events['disease'].str.contains("influenza", case=False, na=False)]
            df_events.to_csv(downloaded_events_file, index=False)
        
        #Run the outbreak download and filtering for bad lines.
        downloaded_outbreaks_file = os.path.join(output_folder, "Influenza_left.csv")
        bad_lines_file = os.path.join(output_folder, "Bad.csv")
        cleaned_csv_file = os.path.join(output_folder, "Processed_BadLines_Influenza.csv")
        cleaned_csv_file_processed = os.path.join(output_folder, "Processed_BadLines_joined_latlong.csv")
        downloaded_outbreaks_removed_bad_lines_file = os.path.join(output_folder, "Influenza_left_bad_lines_removed.csv")
        final_file = os.path.join(output_folder, "combined_file.csv")
        final_merged_file = os.path.join(output_folder, "mapped_combined_file.csv")
        
        self.extract_data(downloaded_events_file, downloaded_outbreaks_file)
        self.extract_bad_lines_with_csv(downloaded_outbreaks_file, bad_lines_file)
        self.create_cleaned_csv(bad_lines_file, cleaned_csv_file)
        self.remove_bad_lines_from_csv(downloaded_outbreaks_file, downloaded_outbreaks_removed_bad_lines_file)
        self.join_long_lat(cleaned_csv_file, cleaned_csv_file_processed)
        self.combine_final(downloaded_outbreaks_removed_bad_lines_file, cleaned_csv_file_processed, final_file)
        self.map_event_id(downloaded_events_file, final_file, final_merged_file)
        self.close()

    '''def run(self):
        os.makedirs('./input', exist_ok=True)
        os.makedirs('./output', exist_ok=True)
        """Run the complete scraping process."""
        website = "https://wahis.woah.org/#/event-management"
        input_folder = "./input"
        output_folder = "./output"
        downloaded_events_file = os.path.join(input_folder, "Wahis_event_id.csv")
        if not os.path.exists(downloaded_events_file):
            self.remove_event_list_files(input_folder)
            self.open_website(website)
            self.close_cookie_popup()
            self.set_items_per_page(website)
            self.get_total_pages()
            self.download_data()
            
            # Double check if Event-list files were downloaded
            event_list_files = glob.glob(os.path.join(input_folder, "Event-list-*.csv"))
            print("Downloaded event files:", event_list_files)
            if not event_list_files:
                 raise FileNotFoundError("No Event-list CSVs were downloaded into ./input.")
            #self.combine_to_csv(input_folder, downloaded_events_file)
            # Filter downloaded_events_file to only rows where 'disease' contains 'influenza' (case-insensitive)
            df_events = pd.read_csv(downloaded_events_file)

            downloaded_events_file = os.path.join(input_folder, "Wahis_event_id.csv")

            # Only download + combine if the file doesn't exist
            if not os.path.exists(downloaded_events_file):
                self.remove_event_list_files(input_folder)
                self.open_website(website)
                self.close_cookie_popup()
                self.set_items_per_page(website)
                self.get_total_pages()
                self.download_data()
                
                # Check if Event-list files were downloaded
                event_list_files = glob.glob(os.path.join(input_folder, "Event-list-*.csv"))
                print("Downloaded event files:", event_list_files)
                if not event_list_files:
                    raise FileNotFoundError("No Event-list CSVs were downloaded into ./input.")
                
                # Combine CSVs into one file
                self.combine_to_csv(input_folder, downloaded_events_file)

                # ➕ Add the influenza filter here — only after file creation
                df_events = pd.read_csv(downloaded_events_file)
                df_events = df_events[df_events['disease'].str.contains("influenza", case=False, na=False)]
                df_events.to_csv(downloaded_events_file, index=False)

        
        #Run the outbreak download and filtering for bad lines.
        downloaded_outbreaks_file = os.path.join(output_folder, "Influenza_left.csv")
        bad_lines_file = os.path.join(output_folder, "Bad.csv")
        cleaned_csv_file = os.path.join(output_folder, "Processed_BadLines_Influenza.csv")
        cleaned_csv_file_processed = os.path.join(output_folder, "Processed_BadLines_joined_latlong.csv")
        downloaded_outbreaks_removed_bad_lines_file = os.path.join(output_folder, "Influenza_left_bad_lines_removed.csv")
        final_file = os.path.join(output_folder, "combined_file.csv")
        final_merged_file = os.path.join(output_folder, "mapped_combined_file.csv")
        
        self.extract_data(downloaded_events_file, downloaded_outbreaks_file)
        self.extract_bad_lines_with_csv(downloaded_outbreaks_file, bad_lines_file)
        self.create_cleaned_csv(bad_lines_file, cleaned_csv_file)
        self.remove_bad_lines_from_csv(downloaded_outbreaks_file, downloaded_outbreaks_removed_bad_lines_file)
        self.join_long_lat(cleaned_csv_file, cleaned_csv_file_processed)
        self.combine_final(downloaded_outbreaks_removed_bad_lines_file, cleaned_csv_file_processed, final_file)
        self.map_event_id(downloaded_events_file, final_file, final_merged_file)
        self.close()'''

    def close(self):
        """Close the browser."""
        self.driver.quit()

# Usage
if __name__ == "__main__":
    scraper = WAHISWebScraper('./input')  # Set Chrome download directory
    try:
        scraper.run()
    finally:
        scraper.close()