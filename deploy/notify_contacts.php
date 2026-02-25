<?php
/**
 * Runs via cron every 5 minutes.
 * Checks contacts.json for new entries and sends email notification.
 */

$contacts_file  = '/home/adloccbvmx/task2vec/contacts.json';
$last_sent_file = '/home/adloccbvmx/task2vec/contacts_last_sent.txt';
$notify_email   = 'jussi.tuominen@softerhr.com';

if (!file_exists($contacts_file)) exit(0);

$last_sent = file_exists($last_sent_file)
    ? (int) trim(file_get_contents($last_sent_file))
    : 0;

$new_entries = [];
foreach (file($contacts_file, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) as $line) {
    $entry = json_decode($line, true);
    if (!$entry) continue;
    $entry_time = strtotime($entry['time'] ?? '');
    if ($entry_time > $last_sent) {
        $new_entries[] = $entry;
    }
}

if (empty($new_entries)) exit(0);

foreach ($new_entries as $e) {
    $subject = 'task2vec contact from ' . $e['name'];
    $body    = "New message via task2vec.com\n\n"
             . "Name:    {$e['name']}\n"
             . "Email:   {$e['email']}\n"
             . "Time:    {$e['time']}\n\n"
             . "Message:\n{$e['message']}\n";
    $headers = "From: noreply@task2vec.com\r\nReply-To: {$e['email']}";
    mail($notify_email, $subject, $body, $headers);
}

file_put_contents($last_sent_file, time());
echo "Notified: " . count($new_entries) . " new submission(s)\n";
