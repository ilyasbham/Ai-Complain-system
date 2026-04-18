package com.example.demo.controller;


import org.springframework.web.bind.annotation.*;
import java.util.Map;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;




@CrossOrigin(origins = "*")
@RestController
@RequestMapping("/api")
public class ComplaintController {

    @PostMapping("/predict")
    public String predict(@RequestBody Map<String, String> request) {

        String text = request.get("text");

        try {
            URL url = new URL("http://127.0.0.1:8000/predict");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);

            String jsonInput = "{\"text\":\"" + text + "\"}";

            OutputStream os = conn.getOutputStream();
            os.write(jsonInput.getBytes());
            os.flush();
            os.close();

            BufferedReader br = new BufferedReader(
                    new InputStreamReader(conn.getInputStream()));

            String response = br.readLine();
            br.close();

            return response;

        } catch (Exception e) {
            return "Error: " + e.getMessage();
        }
    }
}
